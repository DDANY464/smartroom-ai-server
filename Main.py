from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ---- Request from ESP32 ----
class SmartRoomRequest(BaseModel):
    motion: bool
    noise: int
    mood: int
    command: str

# ---- Response back to ESP32 ----
class SmartRoomResponse(BaseModel):
    response: str   # text to speak
    emotion: str    # e.g. "happy", "annoyed"
    action: str     # e.g. "lights_on", "none"


@app.post("/smartroom", response_model=SmartRoomResponse)
def smartroom_ai(req: SmartRoomRequest):
    """
    This is the AI brain endpoint your ESP32 calls.
    For now it's rule-based; you can later swap in a real LLM.
    """

    # --- Simple personality logic (you can make this as deep as you want) ---
    # Base response
    text = "Hey Danny."
    emotion = "neutral"
    action = "none"

    # Motion-based behavior
    if req.motion:
        text = "Welcome back, Danny."
        emotion = "happy"

    # Noise-based behavior
    if req.noise > 60:
        text = "Whoa, that was loud. Take it easy."
        emotion = "annoyed"

    # Command-based behavior
    cmd = req.command.lower().strip()

    if "lights" in cmd and "on" in cmd:
        text = "Turning the lights on for you."
        action = "lights_on"
        emotion = "helpful"

    elif "lights" in cmd and "off" in cmd:
        text = "Turning the lights off. Goodnight."
        action = "lights_off"
        emotion = "calm"

    elif "hello" in cmd or "hi" in cmd:
        text = "Hey Danny, I’m always watching… in a friendly way."
        emotion = "playful"

    # Mood influence (0–100)
    if req.mood < 30:
        text += " You seem a bit low. Want me to chill the lights?"
    elif req.mood > 70:
        text += " You’re in a good mood, I can feel it."

    return SmartRoomResponse(
        response=text,
        emotion=emotion,
        action=action
    )
