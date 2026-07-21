from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

class SmartRoomInput(BaseModel):
    motion: bool
    noise: int
    command: str

@app.post("/smartroom")
async def smartroom(data: SmartRoomInput):
    try:
        action = "none"
        cmd = data.command.lower()

        # LIGHT ON/OFF
        if "lights on" in cmd or "turn on the lights" in cmd or "light on" in cmd:
            action = "lights_on"

        elif "lights off" in cmd or "turn off the lights" in cmd or "light off" in cmd:
            action = "lights_off"

        # COLOR COMMANDS
        colors = {
            "red": "set_color_red",
            "blue": "set_color_blue",
            "green": "set_color_green",
            "purple": "set_color_purple",
            "yellow": "set_color_yellow",
            "white": "set_color_white"
        }

        for color_word, color_action in colors.items():
            if color_word in cmd:
                action = color_action

        # AI reasoning
        messages = [
            {
                "role": "user",
                "content": (
                    f"Motion: {data.motion}\n"
                    f"Noise: {data.noise}\n"
                    f"Command: {data.command}\n"
                    f"Action triggered: {action}\n\n"
                    "Explain the reasoning behind this action and respond as the Smart Room AI."
                )
            }
        ]

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )

        ai_text = response.choices[0].message.content

        return {
            "status": "ok",
            "action": action,
            "ai_response": ai_text
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
