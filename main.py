import os 
import requests
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# Autonomous Memory System (Level 1)
# -------------------------------------------------
conversation_history = []   # short-term memory
memory = {}                 # long-term autonomous memory

def update_memory(user_text):
    text = user_text.lower()

    if "my " in text and " is " in text:
        try:
            key = text.split("my ")[1].split(" is ")[0].strip()
            value = text.split(" is ")[1].strip()
            key = key.replace(" ", "_")
            memory[key] = value
        except:
            pass

    if "i like " in text:
        try:
            value = text.split("i like ")[1].strip()
            memory["likes_" + value.replace(" ", "_")] = True
        except:
            pass

    if "i prefer " in text:
        try:
            value = text.split("i prefer ")[1].strip()
            memory["preference"] = value
        except:
            pass

    if "my dog's name is" in text:
        try:
            name = text.split("my dog's name is")[1].strip()
            memory["dog_name"] = name
        except:
            pass


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
# Environment Variables (Render)
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")


# -------------------------------------------------
# ElevenLabs Debug Endpoint (Corrected)
# -------------------------------------------------
@app.get("/debug-eleven")
def debug_eleven():
    key = os.getenv("ELEVENLABS_API_KEY")  # load at runtime
    return {
        "api_key_is_none": key is None,
        "api_key_length": len(key) if key else 0,
        "api_key_preview": key[:6] if key else "NONE",
        "voice_id": VOICE_ID
    }


# -------------------------------------------------
# ElevenLabs TTS Function (Corrected)
# -------------------------------------------------
def elevenlabs_tts(text):
    api_key = os.getenv("ELEVENLABS_API_KEY")  # load at runtime

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "model_id": "eleven_multilingual_v2",
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.content


# -------------------------------------------------
# Nova Prompt (Your original prompt preserved)
# -------------------------------------------------
NOVA_PROMPT = """
You are Nova, Danny’s Smart Room AI assistant...
(keeping your full prompt exactly as-is)
"""


# -------------------------------------------------
# 1. Nova Audio Endpoint (STT → Nova → TTS)
# -------------------------------------------------
@app.post("/audio")
async def audio_route(request: Request):
    raw_audio = await request.body()

    stt_response = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", raw_audio, "audio/wav")},
        data={"model": "whisper-large-v3"},
        timeout=60
    )
    stt_response.raise_for_status()
    stt_text = stt_response.json().get("text", "")

    update_memory(stt_text)
    memory_text = f"Known facts: {memory}"

    messages = [
        {"role": "system", "content": NOVA_PROMPT + "\n" + memory_text},
        *conversation_history,
        {"role": "user", "content": stt_text}
    ]

    nova_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages},
        timeout=60
    )
    nova_response.raise_for_status()
    nova_reply = nova_response.json()["choices"][0]["message"]["content"]

    conversation_history.append({"role": "user", "content": stt_text})
    conversation_history.append({"role": "assistant", "content": nova_reply})

    audio_bytes = elevenlabs_tts(nova_reply)

    return Response(content=audio_bytes, media_type="audio/mpeg")


# -------------------------------------------------
# 2. Nova Text Endpoint
# -------------------------------------------------
@app.post("/nova")
async def nova_route(request: Request):
    data = await request.json()
    user_text = data.get("text", "")

    update_memory(user_text)
    memory_text = f"Known facts: {memory}"

    messages = [
        {"role": "system", "content": NOVA_PROMPT + "\n" + memory_text},
        *conversation_history,
        {"role": "user", "content": user_text}
    ]

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages},
        timeout=60
    )

    ai_text = response.json()["choices"][0]["message"]["content"]

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": ai_text})

    return {"response": ai_text}


# -------------------------------------------------
# 3. Nova Speak (Direct Text → TTS)
# -------------------------------------------------
@app.post("/nova/speak")
async def nova_speak(request: Request):
    data = await request.json()
    text = data.get("text", "")

    audio_bytes = elevenlabs_tts(text)

    return Response(content=audio_bytes, media_type="audio/mpeg")


# -------------------------------------------------
# Render Port Binding
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

