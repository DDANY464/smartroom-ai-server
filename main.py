from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# -----------------------------
# Request / Response Models
# -----------------------------
class SmartRoomRequest(BaseModel):
    motion: bool = False
    noise: int = 0
    mood: int = 50
    command: str = ""

class SmartRoomResponse(BaseModel):
    response: str
    emotion: str
    action: str

# -----------------------------
# Groq AI Function
# -----------------------------
def ask_groq(prompt: str) -> str:
    import requests
    import os

    GROQ_KEY = os.getenv("GROQ_API_KEY")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_KEY}"
    }

    data = {
        "model": "llama3-8b",
        "messages": [
            {"role": "system", "content": "You are Danny's Smart Room AI assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        r = requests.post(url, json=data, headers=headers)
        result = r.json()

        # Groq returns: result["choices"][0]["message"]["content"]
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]

        return f"AI error: {result}"
    except Exception as e:
        return f"AI error: {e}"# -----------------------------
# Main Smart Room AI Brain
# -----------------------------
@app.post("/smartroom", response_model=SmartRoomResponse)
def smartroom_ai(req: SmartRoomRequest):

    text = "Hey Danny."
    emotion = "neutral"
    action = "none"

    cmd = (req.command or "").lower().strip()

    # -----------------------------
    # SENSOR FUSION
    # -----------------------------
    if req.motion:
        text = "Welcome back, Danny."
        emotion = "happy"

    if req.noise > 60:
        text = "Whoa, that was loud."
        emotion = "annoyed"

    # -----------------------------
    # ACTION COMMANDS
    # -----------------------------
    if "lights" in cmd and "on" in cmd:
        text = "Turning the lights on."
        action = "lights_on"
        emotion = "helpful"

    elif "lights" in cmd and "off" in cmd:
        text = "Turning the lights off."
        action = "lights_off"
        emotion = "calm"

    elif "hello" in cmd or "hi" in cmd:
        text = "Hey Danny, I'm online."
        emotion = "playful"

    # -----------------------------
    # PERSONALITY QUESTIONS
    # -----------------------------
    elif "how's your day" in cmd or "how is your day" in cmd:
        text = "My day is great — lots of sensors to watch."
        emotion = "happy"

    elif "joke" in cmd:
        text = "Why did the ESP32 cross the road? To get better WiFi."
        emotion = "funny"

    # -----------------------------
    # GROQ AI FALLBACK
    # -----------------------------
    elif cmd != "":
        text = ask_groq(cmd)
        emotion = "informative"

    # -----------------------------
    # MOOD INFLUENCE
    # -----------------------------
    if req.mood < 30:
        text += " You seem a bit low today."
    elif req.mood > 70:
        text += " I can feel your good mood."

    return SmartRoomResponse(
        response=text,
        emotion=emotion,
        action=action
    )