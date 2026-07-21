# ──────────────────────────────────────────────────────────────────────────────
#  smartroom_api.py
# ──────────────────────────────────────────────────────────────────────────────
"""
FastAPI service that receives a sensor payload + a free‑form voice command,
decides what the room should do (lights, colour, …) and asks Groq’s LLM to
explain the reasoning.
"""

import os
import logging
from typing import Literal, Optional

import anyio
import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, conint, validator

# --------------------------------------------------------------------------- #
#   Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("smartroom")

# --------------------------------------------------------------------------- #
#   Environment & Groq client
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("🔑 Environment variable GROQ_API_KEY is required")

from groq import Groq
_sync_client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------------------------------- #
#   FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Smart‑Room AI",
    version="1.0.0",
    description="Parse voice commands & sensor data, decide an action, and let Groq explain it.",
)

# --------------------------------------------------------------------------- #
#   Pydantic models
# --------------------------------------------------------------------------- #
class SmartRoomInput(BaseModel):
    motion: bool = Field(..., description="True if motion was detected")
    noise: conint(ge=0, le=120) = Field(..., description="Ambient noise level in dB")
    command: str = Field(..., min_length=1, description="Free‑form voice command")

    @validator("command")
    def _strip(cls, v: str) -> str:
        return v.strip()


class SmartRoomOutput(BaseModel):
    status: Literal["ok", "error"]
    action: Optional[str] = None
    ai_response: Optional[str] = None
    detail: Optional[str] = None


# --------------------------------------------------------------------------- #
#   Helper utilities
# --------------------------------------------------------------------------- #
class ActionResult:
    def __init__(self, lights: Optional[str] = None, colour: Optional[str] = None):
        self.lights = lights
        self.colour = colour

    @property
    def combined(self) -> str:
        parts = [self.lights, self.colour]
        return "+".join(filter(None, parts)) or "none"


def parse_command(raw: str) -> ActionResult:
    cmd = raw.lower()

    # Lights
    lights_action = None
    if any(kw in cmd for kw in ("lights on", "turn on the lights", "light on", "illuminate")):
        lights_action = "lights_on"
    elif any(kw in cmd for kw in ("lights off", "turn off the lights", "light off", "darken")):
        lights_action = "lights_off"

    # Colors
    colour_map = {
        "red": "set_color_red",
        "blue": "set_color_blue",
        "green": "set_color_green",
        "purple": "set_color_purple",
        "yellow": "set_color_yellow",
        "white": "set_color_white",
        "orange": "set_color_orange",
        "pink": "set_color_pink",
    }

    colour_action = None
    for word, act in colour_map.items():
        if word in cmd:
            colour_action = act
            break

    return ActionResult(lights=lights_action, colour=colour_action)


# --------------------------------------------------------------------------- #
#   Groq helper – retries + thread‑pool execution
# --------------------------------------------------------------------------- #
async def _call_groq_llm(messages: list[dict]) -> str:
    max_tries = 3
    backoff = 1.0

    async def _sync_call() -> str:
        response = _sync_client.chat.completions.create(
            model="llama3.1-8b-instant",
            messages=messages,
        )
        if not getattr(response, "choices", None):
            raise RuntimeError("Groq returned an empty response")
        return response.choices[0].message.content

    for attempt in range(1, max_tries + 1):
        try:
            return await anyio.to_thread.run_sync(_sync_call)
        except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {
                429, 502, 503, 504
            }:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Groq API error: {exc.response.status_code}",
                )
            if attempt == max_tries:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Groq service unavailable after several retries",
                )
            log.warning(
                "Groq request failed (attempt %s/%s) – retrying in %.1fs: %s",
                attempt,
                max_tries,
                backoff,
                exc,
            )
            await anyio.sleep(backoff)
            backoff *= 2


# --------------------------------------------------------------------------- #
#   Endpoint (THIS WAS MISSING — NOW FIXED)
# --------------------------------------------------------------------------- #
@app.post(
    "/smartroom",
    response_model=SmartRoomOutput,
    status_code=status.HTTP_200_OK,
)
async def smartroom_endpoint(payload: SmartRoomInput) -> SmartRoomOutput:
    parsed = parse_command(payload.command)
    action = parsed.combined
    log.info("Parsed command → %s", action)

    messages = [
        {
            "role": "user",
            "content": (
                f"Motion: {payload.motion}\n"
                f"Noise: {payload.noise}\n"
                f"Command: {payload.command}\n"
                f"Action chosen: {action}\n\n"
                "Explain the reasoning behind this action as the Smart‑Room AI."
            ),
        }
    ]

    try:
        ai_text = await _call_groq_llm(messages)
    except HTTPException as exc:
        raise exc
    except Exception as exc:
        log.exception("Unexpected Groq error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected Groq error: {exc}",
        )

    return SmartRoomOutput(
        status="ok",
        action=action,
        ai_response=ai_text,
    )
