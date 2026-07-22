import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, validator
from typing import Literal, Optional
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
log = logging.getLogger("smartroom")

# Cloud AI client
client = OpenAI(api_key="YOUR_API_KEY")

app = FastAPI(
    title="Smart‑Room AI (Cloud Model)",
    version="1.0.0",
    description="Smart Room automation powered by cloud inference.",
)

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

async def call_ai(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"AI error: {exc}"
        )

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

    ai_text = await call_ai(prompt)

    return SmartRoomOutput(
        status="ok",
        action=action,
        ai_response=ai_text,
    )
