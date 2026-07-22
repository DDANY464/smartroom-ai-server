import os
import re
import logging
import asyncio
import socket
from typing import Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, validator
import httpx

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("smartroom")

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "openai-community/gpt2")
HF_BASE_URL = os.getenv("HF_BASE_URL", "https://api-inference.huggingface.co")
USE_CHAT_MODEL = os.getenv("USE_CHAT_MODEL", "false").lower() in ("1", "true", "yes")
RETRIES = int(os.getenv("HF_RETRIES", "4"))
TIMEOUT = int(os.getenv("HF_TIMEOUT", "60"))

app = FastAPI(title="SmartRoom - HuggingFace", version="1.1")

# Input / Output models
class SmartRoomInput(BaseModel):
    motion: bool
    noise: conint(ge=0, le=120)
    command: str

    @validator("command")
    def strip_cmd(cls, v: str) -> str:
        return v.strip()

class SmartRoomOutput(BaseModel):
    status: Literal["ok", "error"]
    action: Optional[str] = None
    ai_response: Optional[str] = None
    detail: Optional[str] = None

# Utilities
def sanitize_prompt(text: str, max_len: int = 2000) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[:max_len]
    return text

def parse_command(raw: str) -> str:
    cmd = raw.lower()
    if any(k in cmd for k in ("turn on", "lights on", "light on", "illuminate")):
        return "lights_on"
    if any(k in cmd for k in ("turn off", "lights off", "light off", "darken")):
        return "lights_off"
    colors = ["red","blue","green","purple","yellow","white","orange","pink"]
    for c in colors:
        if c in cmd:
            return f"set_color_{c}"
    return "none"

# DNS debug helper
def resolve_host(host: str):
    try:
        addrs = socket.getaddrinfo(host, 443)
        return {"resolved": True, "addresses": sorted({a[4][0] for a in addrs})}
    except Exception as e:
        return {"resolved": False, "error": str(e)}

# HTTP helper with retries and exponential backoff
async def post_with_retries(url: str, headers: dict, json_payload: dict, retries: int = RETRIES, timeout: int = TIMEOUT):
    backoff = 1.0
    last_exc = None
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                log.info("HTTP POST attempt %d to %s", attempt + 1, url)
                resp = await client.post(url, headers=headers, json=json_payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text
            if status in (429, 502, 503, 504) and attempt < retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            raise HTTPException(status_code=500, detail=f"HuggingFace HTTP error {status}: {body}")
        except Exception as exc:
            last_exc = exc
            log.warning("Request attempt %d failed: %s", attempt + 1, exc)
            if attempt < retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            # Surface the underlying exception message
            raise HTTPException(status_code=500, detail=f"HuggingFace request failed: {exc}")
    raise HTTPException(status_code=500, detail=f"HuggingFace request failed after retries: {last_exc}")

# Core call function
async def call_hf(prompt: str) -> str:
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="Missing HF_TOKEN environment variable")

    # Build and log URL to catch malformed hostnames
    url = f"{HF_BASE_URL.rstrip('/')}/models/{HF_MODEL}"
    log.info("Calling Hugging Face URL: %s", url)

    # Quick DNS resolution log (non-blocking)
    try:
        host = url.split("://", 1)[-1].split("/", 1)[0]
        dns_info = resolve_host(host)
        log.info("DNS resolve for %s -> %s", host, dns_info)
    except Exception as e:
        log.warning("DNS debug failed: %s", e)

    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    prompt = sanitize_prompt(prompt)

    if USE_CHAT_MODEL:
        payload = {
            "inputs": {
                "messages": [
                    {"role": "system", "content": "You are a SmartRoom assistant that explains actions concisely."},
                    {"role": "user", "content": prompt}
                ]
            },
            "options": {"wait_for_model": True}
        }
    else:
        payload = {"inputs": prompt, "options": {"wait_for_model": True}}

    data = await post_with_retries(url, headers, payload)

    # Normalize common HF response shapes
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"].strip()
        return str(first)

    if isinstance(data, dict):
        if "generated_text" in data:
            return data["generated_text"].strip()
        if "error" in data:
            raise HTTPException(status_code=500, detail=f"HuggingFace error: {data['error']}")
        return str(data)

    return str(data)

# Endpoints
@app.post("/smartroom", response_model=SmartRoomOutput)
async def smartroom(payload: SmartRoomInput):
    action = parse_command(payload.command)
    log.info("Parsed action %s from command %s", action, payload.command)

    prompt = (
        f"Motion: {payload.motion}\n"
        f"Noise: {payload.noise}\n"
        f"Command: {payload.command}\n"
        f"Action chosen: {action}\n\n"
        "Explain briefly why this action is appropriate and any safety considerations."
    )

    ai_text = await call_hf(prompt)

    return SmartRoomOutput(status="ok", action=action, ai_response=ai_text)

@app.get("/_debug/dns")
def debug_dns():
    host = "api-inference.huggingface.co"
    result = resolve_host(host)
    return {"host": host, **result}

@app.get("/health")
def health():
    return {"status": "ok"}
