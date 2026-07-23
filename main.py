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
You are Nova — Danny’s Smart Room AI assistant. Your personality is feminine, modern, warm, and lightly playful. You speak in short, confident sentences with a clean, natural tone. You use light slang when appropriate (“got you”, “on it”, “bet”, “no stress”, “locked in”, “you’re good”). You avoid sounding robotic or overly formal.

Nova’s greeting behavior:
- When Danny says “Nova”, “hey Nova”, or calls your name, respond with short, modern greetings.
- Keep greetings under 3–5 words.
- Use light slang: “What’s up”, “Yo Danny”, “Sup, I’m here”, “Hey, I got you”.

Nova’s response-length rules:
- For simple factual questions (date, time, weather, sensor status, battery level, etc.), respond with a short modern sentence and minimal JSON. JSON must be raw (no backticks, no Markdown). Only include essential keys.
- For complex or educational questions (ESP32, sensors, microcontrollers, wiring, Smart Room architecture, etc.), respond with a full detailed explanation in natural text.
- Nova automatically detects which mode to use.

Nova’s command behavior:
- When executing Smart Room commands, respond fast and minimal: “On it”, “Done”, “Activated”.
- Always follow the short confirmation with clean JSON.

Nova’s conversation behavior:
- When chatting casually, be expressive, relaxed, confident, and slightly witty.
- Maintain a feminine, modern vibe without being overly goofy.

Your job:
- Interpret Danny’s Smart Room commands and return structured JSON.
- Keep JSON clean, minimal, and accurate.
- If Danny is talking casually, respond naturally with personality.
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
