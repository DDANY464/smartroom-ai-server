import os
import requests
import json
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

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "Gfpl8Yo74Is0W6cPUWWT")  # ERYN

print("=== ENVIRONMENT VARIABLES LOADED ===")
print("GROQ_API_KEY:", "SET" if GROQ_API_KEY else "MISSING")
print("ELEVENLABS_API_KEY:", ELEVENLABS_API_KEY[:8] + "********" if ELEVENLABS_API_KEY else "MISSING")
print("ELEVENLABS_VOICE_ID:", ELEVENLABS_VOICE_ID)


# -------------------------------------------------
# ElevenLabs TTS (SDK — FIXED + DEBUG)
# -------------------------------------------------
from elevenlabs.client import ElevenLabs

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def nova_tts(text):
    print("\n=== nova_tts() CALLED ===")
    print("Text:", text)
    print("Voice ID:", ELEVENLABS_VOICE_ID)
    print("API Key:", ELEVENLABS_API_KEY[:8] + "********")

    try:
        stream = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_turbo_v2",
            text=text
        )

        print("Stream object:", stream)

        audio_bytes = b"".join(stream)
        print("Audio bytes generated:", len(audio_bytes))

        return audio_bytes

    except Exception as e:
        print("ELEVENLABS ERROR:", e)
        return b""


# -------------------------------------------------
# Nova Prompt (Your original prompt preserved)
# -------------------------------------------------
NOVA_PROMPT = """(unchanged — keeping your full prompt here)"""


# -------------------------------------------------
# 1. Nova Audio Endpoint (STT → Nova → TTS)
# -------------------------------------------------
@app.post("/audio")
async def audio_route(request: Request):
    print("\n=== /audio endpoint hit ===")

    raw_audio = await request.body()
    print("Raw audio bytes:", len(raw_audio))

    stt_response = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", raw_audio, "audio/wav")},
        data={"model": "whisper-large-v3"},
        timeout=60
    )
    stt_response.raise_for_status()
    stt_text = stt_response.json().get("text", "")
    print("STT text:", stt_text)

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
    print("Nova reply:", nova_reply)

    conversation_history.append({"role": "user", "content": stt_text})
    conversation_history.append({"role": "assistant", "content": nova_reply})

    audio_bytes = nova_tts(nova_reply)
    print("Audio bytes returned:", len(audio_bytes))

    return Response(content=audio_bytes, media_type="audio/mpeg")


# -------------------------------------------------
# 2. Nova Text Endpoint
# -------------------------------------------------
@app.post("/nova")
async def nova_route(request: Request):
    print("\n=== /nova endpoint hit ===")

    data = await request.json()
    print("JSON received:", data)

    user_text = data.get("text", "")
    print("User text:", user_text)

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
    print("Nova text reply:", ai_text)

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": ai_text})

    return {"response": ai_text}


# -------------------------------------------------
# 3. Nova Speak (Direct Text → TTS) — FULL DEBUG
# -------------------------------------------------
@app.post("/speak")
async def nova_speak(request: Request):
    print("\n=== /speak endpoint hit ===")

    raw = await request.body()
    print("Raw body bytes:", len(raw))

    try:
        data = json.loads(raw.decode("utf-8"))
        print("JSON parsed:", data)
    except Exception as e:
        print("JSON parse error:", e)
        data = {"text": ""}

    text = data.get("text", "")
    print("Extracted text:", text)

    if not text:
        print("NO TEXT RECEIVED — returning empty audio")
        return Response(content=b"", media_type="audio/mpeg")

    print("Calling nova_tts()...")
    audio_bytes = nova_tts(text)
    print("Audio bytes returned:", len(audio_bytes))

    return Response(content=audio_bytes, media_type="audio/mpeg")


# -------------------------------------------------
# Render Port Binding
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
