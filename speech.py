import os
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

@app.post("/speech")
async def speech_endpoint(request: Request):
    raw_audio = await request.body()

    # -------------------------------------------------
    # 1. Speech-to-Text (Groq Whisper)
    # -------------------------------------------------
    stt_text = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", raw_audio, "audio/wav")},
        data={"model": "whisper-large-v3"}
    ).json()["text"]

    # -------------------------------------------------
    # 2. Nova Brain (Groq LLM)
    # -------------------------------------------------
    nova_reply = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are Nova, the Smart Room AI assistant."},
                {"role": "user", "content": stt_text}
            ]
        }
    ).json()["choices"][0]["message"]["content"]

    # -------------------------------------------------
    # 3. ElevenLabs TTS (Nova → Voice)
    # -------------------------------------------------
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": nova_reply,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.8
        }
    }

    tts_audio = requests.post(tts_url, headers=headers, json=payload).content

    # Return raw audio bytes to ESP32
    return tts_audio
