import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, validator
from typing import Optional, Literal
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("smartroom")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "gpt2")  # change to your chosen model id

app = FastAPI(title="SmartRoom - HuggingFace", version="1.0")

class SmartRoomInput(BaseModel):
    motion: bool
    noise: conint(ge=0, le=120)
    command: str
    @validator("command")
    def strip_cmd(cls, v): return v.strip()

class SmartRoomOutput(BaseModel):
    status: Literal["ok","error"]
    action: Optional[str] = None
    ai_response: Optional[str] = None
    detail: Optional[str] = None

async def call_hf(prompt: str) -> str:
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN")
    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Many HF models return a list with generated_text or a dict; handle common shapes:
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"]
                return str(first)
            if isinstance(data, dict):
                # some models return {"error": "..."} or {"generated_text": "..."}
                if "error" in data:
                    raise HTTPException(status_code=500, detail=data["error"])
                if "generated_text" in data:
                    return data["generated_text"]
            return str(data)
    except httpx.HTTPStatusError as e:
        # surface HF error message
        detail = e.response.text
        raise HTTPException(status_code=500, detail=f"HuggingFace error: {detail}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"HuggingFace request failed: {exc}")

def parse_command(raw: str) -> str:
    cmd = raw.lower()
    if "turn on" in cmd or "lights on" in cmd:
        return "lights_on"
    if "turn off" in cmd or "lights off" in cmd:
        return "lights_off"
    return "none"

@app.post("/smartroom", response_model=SmartRoomOutput)
async def smartroom(payload: SmartRoomInput):
    action = parse_command(payload.command)
    prompt = f"Motion: {payload.motion}\nNoise: {payload.noise}\nCommand: {payload.command}\nAction: {action}\nExplain the reasoning briefly."
    ai_text = await call_hf(prompt)
    return SmartRoomOutput(status="ok", action=action, ai_response=ai_text)
