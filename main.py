# ──────────────────────────────────────────────────────────────────────────────
#  main.py — Smart Room AI using HuggingFace Universal Inference API (FREE)
# ──────────────────────────────────────────────────────────────────────────────

import os
import logging
from typing import Literal, Optional

import anyio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, validator

# --------------------------------------------------------------------------- #
#   Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("smartroom")

# --------------------------------------------------------------------------- #
#   Environment
# --------------------------------------------------------------------------- #
HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY environment variable is required")

# Render-friendly HuggingFace endpoint
HF_MODEL_URL = "https://huggingface.co/api/InferenceApi"

# --------------------------------------------------------------------------- #
#   FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Smart‑Room AI (HuggingFace Universal API)",
    version="1.0.0",
    description="Smart Room automation powered by HuggingFace Inference API.",
)

# --------------------------------------------------------------------------- #
#   Pydantic models
# --------------------------------------------------------------------------- #
class SmartRoomInput(BaseModel):
    motion: bool
    noise: conint(ge=0, le=120)
    command: str

    @validator("command")
    def strip_cmd(cls, v):
        return v.strip()


class SmartRoomOutput(BaseModel):
    status: Literal["ok", "error"]
    action: Optional[str] = None
    ai_response: Optional[str] = None
    detail: Optional[str] = None


# --------------------------------------------------------------------------- #
#   Command Parser
# --------------------------------------------------------------------------- #
class ActionResult:
    def __init__(self, lights=None, colour=None):
        self.lights = lights
        self.colour = colour

    @property
    def combined(self):
        parts = [self.lights, self.colour]
        return "+".join(filter(None, parts)) or "none"


def parse_command(raw: str) -> ActionResult:
    cmd = raw.lower()

    lights = None
    if any(kw in cmd for kw in ("lights on", "turn on the lights", "light on", "illuminate")):
        lights = "lights_on"
    elif any(kw in cmd for kw in ("lights off", "turn off the lights", "light off", "darken")):
        lights = "lights_off"

    colors = {
        "red": "set_color_red",
        "blue": "set_color_blue",
        "green": "set_color_green",
        "purple": "set_color_purple",
        "yellow": "set_color_yellow",
        "white": "set_color_white",
        "orange": "set_color_orange",
        "pink": "set_color_pink",
    }

    colour = None
    for word, act in colors.items():
        if word in cmd:
            colour = act
            break

    return ActionResult(lights, colour)


# --------------------------------------------------------------------------- #
#   HuggingFace Universal API call (Render-friendly)
# --------------------------------------------------------------------------- #
async def call_huggingface(prompt: str) -> str:
    max_tries = 3
    backoff = 1.0

    async def _do_call():
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                HF_MODEL_URL,
                headers={
                    "Authorization": f"Bearer {HF_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "meta-llama/Llama-3.2-1B-Instruct",
                    "inputs": prompt
                },
            )
            response.raise_for_status()
            data = response.json()

            # Universal API returns dict with "generated_text"
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]

            return str(data)

    for attempt in range(1, max_tries + 1):
        try:
            return await _do_call()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {429, 502, 503, 504}:
                raise HTTPException(
                    status_code=502,
                    detail=f"HuggingFace API error: {exc.response.status_code}",
                )
            if attempt == max_tries:
                raise HTTPException(
                    status_code=504,
                    detail="HuggingFace unavailable after retries",
                )
            log.warning(
                "HF request failed (attempt %s/%s) – retrying in %.1fs",
                attempt,
                max_tries,
                backoff,
            )
            await anyio.sleep(backoff)
            backoff *= 2


# --------------------------------------------------------------------------- #
#   Endpoint
# --------------------------------------------------------------------------- #
@app.post("/smartroom", response_model=SmartRoomOutput)
async def smartroom_endpoint(payload: SmartRoomInput) -> SmartRoomOutput:
    parsed = parse_command(payload.command)
    action = parsed.combined
    log.info("Parsed command → %s", action)

    prompt = (
        f"Motion: {payload.motion}\n"
        f"Noise: {payload.noise}\n"
        f"Command: {payload.command}\n"
        f"Action chosen: {action}\n\n"
        "Explain the reasoning behind this action as the Smart‑Room AI."
    )

    try:
        ai_text = await call_huggingface(prompt)
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        log.exception("Unexpected HuggingFace error")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected HuggingFace error: {exc}",
        )

    return SmartRoomOutput(
        status="ok",
        action=action,
        ai_response=ai_text,
    )
