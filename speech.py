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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
TTS_API = os.getenv("TTS_API")  # your text-to-speech model

@app.post("/speech")
async def speech_endpoint(request: Request):
    raw_audio = await request.body()

    # 1. Speech-to-text
    stt_text = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", raw_audio, "audio/wav")},
        data={"model": "whisper-large-v3"}
    ).json()["text"]

    # 2. Send text to Nova brain
    nova_reply = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are Nova..."},
                {"role": "user", "content": stt_text}
            ]
        }
    ).json()["choices"][0]["message"]["content"]

    # 3. Convert Nova’s reply to speech
    tts_audio = requests.post(
        TTS_API,
        json={"text": nova_reply}
    ).content

    return tts_audio
