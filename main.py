import os
import requests
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# FastAPI App Setup
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

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

AI_SPEECH_URL = os.getenv("AI_SPEECH_URL")  # optional if you use another STT model


# -------------------------------------------------
# Nova Personality Prompt (RESTORED)
# -------------------------------------------------
NOVA_PROMPT = """
You are Nova, the Smart Room AI assistant created by Danny.
You are warm, expressive, playful, and extremely helpful.
You speak like a friendly futuristic companion with personality.
You understand Danny’s environment, sensors, radar, ESP32 devices, and his Smart Room project.
You keep responses short, natural, and conversational unless asked otherwise.
You never act robotic — you act alive.
"""


# -------------------------------------------------
# 1. Speech → Text → Nova → ElevenLabs Voice
# -------------------------------------------------
@app.post("/speech")
async def speech_endpoint(request: Request):
    raw_audio = await request.body()

    # 1. Speech-to-text (Groq Whisper)
    stt_response = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", raw_audio, "audio/wav")},
        data={"model": "whisper-large-v3"}
    )

    stt_text = stt_response.json().get("text", "")

    # 2. Nova brain (Groq LLM)
    nova_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": NOVA_PROMPT},
                {"role": "user", "content": stt_text}
            ]
        }
    )

    nova_reply = nova_response.json()["choices"][0]["message"]["content"]

    # 3. ElevenLabs TTS
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

    return Response(content=tts_audio, media_type="audio/wav")


# -------------------------------------------------
# 2. Nova Text Endpoint (no audio)
# -------------------------------------------------
@app.post("/nova")
async def nova_route(request: Request):
    data = await request.json()
    user_text = data.get("text", "")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": NOVA_PROMPT},
                {"role": "user", "content": user_text}
            ]
        }
    )

    ai_text = response.json()["choices"][0]["message"]["content"]
    return {"response": ai_text}


# -------------------------------------------------
# 3. ElevenLabs Direct TTS Endpoint (text → audio)
# -------------------------------------------------
@app.post("/nova/speak")
async def nova_speak(request: Request):
    data = await request.json()
    text = data.get("text", "")

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

    audio_bytes = requests.post(tts_url, headers=headers, json=payload).content

    return Response(content=audio_bytes, media_type="audio/wav")


# -------------------------------------------------
# 4. Railway Port Binding
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
