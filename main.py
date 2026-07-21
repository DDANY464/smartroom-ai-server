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
        parts