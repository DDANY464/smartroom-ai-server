import os
import logging
import requests
import numpy as np
import sounddevice as sd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova-debug-backend")

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Environment Variables (Railway)
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
AI_SPEECH_URL = os.getenv("AI_SPEECH_URL")

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

# -------------------------------------------------
# Nova Personality Endpoint
# -------------------------------------------------
@app.post("/nova")
async def nova_route(request: Request):
    data = await request.json()
    user_text = data.get("text", "")

    logger.info(f"Nova personality request: {user_text}")

    groq_payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are Nova, the Smart Room AI."},
            {"role": "user", "content": user_text}
        ]
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json=groq_payload
    )

    ai_text = response.json()["choices"][0]["message"]["content"]
    logger.info(f"Nova response: {ai_text}")

    return {"response": ai_text}

# -------------------------------------------------
# Smart Room Audio Endpoint (ESP32 → Backend → TTS → JBL)
# -------------------------------------------------
@app.post("/smartroom/audio")
async def smartroom_audio(request: Request):
    logger.info("Incoming Smart Room audio request")

    raw = await request.body()

    radar_presence = request.headers.get("X-Radar-Presence", "unknown")
    logger.info(f"Radar presence: {radar_presence}")

    # Forward audio to your speech model (AI_SPEECH_URL)
    try:
        speech_response = requests.post(
            AI_SPEECH_URL,
            headers={"Content-Type": "application/octet-stream"},
            data=raw
        )

        if speech_response.status_code != 200:
            logger.error(f"Speech model error: {speech_response.text}")
            return {"error": "Speech model failed"}

        audio_data = np.frombuffer(speech_response.content, dtype=np.int16)

        sd.play(audio_data, samplerate=24000)
        sd.wait()

        return {"status": "played"}

    except Exception as e:
        logger.error(f"Audio pipeline error: {e}")
        return {"error": "Audio pipeline crashed"}

# -------------------------------------------------
# ElevenLabs TTS Endpoint (Nova → Voice)
# -------------------------------------------------
@app.post("/nova/speak")
async def nova_speak(request: Request):
    data = await request.json()
    text = data.get("text", "")

    logger.info(f"Nova TTS request: {text}")

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8
        }
    }

    response = requests.post(tts_url, headers=headers, json=payload)

    if response.status_code != 200:
        logger.error(f"ElevenLabs error: {response.text}")
        return {"error": "TTS failed"}

    audio_bytes = response.content

    return audio_bytes

# -------------------------------------------------
# Railway Port Binding
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"DEBUG: Starting Uvicorn on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
