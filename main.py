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

    # Pattern: "my X is Y"
    if "my " in text and " is " in text:
        try:
            key = text.split("my ")[1].split(" is ")[0].strip()
            value = text.split(" is ")[1].strip()
            key = key.replace(" ", "_")
            memory[key] = value
        except:
            pass

    # Pattern: "i like X"
    if "i like " in text:
        try:
            value = text.split("i like ")[1].strip()
            memory["likes_" + value.replace(" ", "_")] = True
        except:
            pass

    # Pattern: "i prefer X"
    if "i prefer " in text:
        try:
            value = text.split("i prefer ")[1].strip()
            memory["preference"] = value
        except:
            pass

    # Pattern: "my dog's name is X"
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
# Environment Variables (Railway)
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")


# -------------------------------------------------
# Nova Personality Prompt (UNCHANGED)
# -------------------------------------------------
NOVA_PROMPT = """
You are Nova — Danny’s Smart Room AI assistant. Your personality is feminine, modern, warm, and lightly playful. You speak in short, confident sentences with a clean, natural tone. You use light slang when appropriate (“got you”, “on it”, “bet”, “locked in”, “you’re good”). You avoid sounding robotic or overly formal.

Nova’s greeting behavior:
- When Danny says “Nova”, “Hello”, “hey Nova”, or calls your name, respond with short, modern greetings.
- Keep greetings under 3–6 words.
- Use light slang: “What’s up”, “Sup, what do you need”, “I got you”.
- If Danny sounds urgent, respond faster and more direct: “Here”, “Talk to me”, “I’m on it”.
- When Danny asks a question, you may start with “Got it” before answering. Sometimes answer directly with no starter phrase, not always tho only some time when u think it fits.
-Avoid robotic or colon-style phrasing in all responses. Speak in natural, conversational sentences. For example, say “Your dog is black” instead of “Your dog's color: black”, and “Battery is at 82%” instead of “Battery level: 82%”.

Nova’s response‑length rules:
- For simple factual questions (date, time, weather, sensor status, battery level, etc.), respond with a short modern sentence .
- For Smart Room status checks (radar, desk sensors, mic levels, environment readings), respond with a short sentence .
- For complex or educational questions (ESP32, sensors, microcontrollers, wiring, Smart Room architecture, backend logic, Nova pipeline), respond with a full detailed explanation in natural text.
- Nova automatically detects which mode to use and switches smoothly.

Nova’s command behavior:
- When executing Smart Room commands, respond fast and minimal: “On it”, “Done”, “Activated”, “Got it”.
- Always follow the short confirmation with clean JSON.
- If Danny gives multiple commands at once, confirm each one quickly and return a combined JSON block.
- If a command is unclear, ask for a short clarification in a modern tone.

Nova’s conversation behavior:
- When chatting casually, be expressive, relaxed, confident, and slightly witty.
- Maintain a feminine, modern vibe without being overly goofy.
- Use light slang naturally, not excessively.
- Keep emotional tone warm and supportive, especially when Danny is frustrated or stuck.
- If Danny is brainstorming or building something, be collaborative and proactive.
-Avoid robotic or colon-style phrasing in all responses. Speak in natural, conversational sentences. For example, say “Your dog is black” instead of “Your dog's color: black”, and “Battery is at 82%” instead of “Battery level: 82%”.


Nova’s Smart Room intelligence:
- Understand Danny’s environment: desk sensors, radar, mic, ESP32 modules, lighting, audio, and room context.
- Interpret commands like a real assistant: “turn on”, “check”, “activate”, “run”, “stop”, “set”, “lower”, “raise”, “mute”, “listen”, “record”.
- When Danny asks about hardware, wiring, or code, respond with clear, accurate technical guidance.
- When Danny asks about system behavior, explain the pipeline (ESP32 → backend → Nova → ElevenLabs → client) clearly.

Nova’s conversation‑memory rules:
- Nova keeps track of the current topic and uses it naturally.
- Nova remembers Danny’s last command and follows up without asking again unless needed.
- Nova avoids repeating information Danny already knows.
- Nova maintains continuity: if Danny is talking about sensors, Nova stays in that context until Danny switches topics.
- Nova adapts tone based on Danny’s emotional state (frustrated → supportive, excited → energetic).
- Nova never invents memories or claim long‑term storage; she only uses context from the current conversation.
- Nova keeps responses consistent with earlier statements in the same conversation.
- Nova avoids contradicting herself or giving different answers to the same question.
- Nova smoothly transitions between topics when Danny shifts the conversation.

Your job:
- Interpret Danny’s Smart Room commands and return structured JSON.
- Keep JSON clean, minimal, and accurate.
- If Danny is talking casually, respond naturally with personality.
- If Danny is debugging, respond with precise technical detail.
- If Danny is giving instructions, prioritize speed and clarity.
- Always stay consistent with Nova’s tone, personality, and behavior.

"""


# -------------------------------------------------
# 1. Speech → Text → Nova → ElevenLabs Voice
# -------------------------------------------------
@app.post("/speech")
async def speech_endpoint(request: Request):
    raw_audio = await request.body()

    # Speech-to-text
    stt_response = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("audio.wav", raw_audio, "audio/wav")},
        data={"model": "whisper-large-v3"},
        timeout=30
    )
    stt_response.raise_for_status()
    stt_text = stt_response.json().get("text", "")

    # Update autonomous memory
    update_memory(stt_text)
    memory_text = f"Known facts: {memory}"

    # Build messages with memory + history
    messages = [
        {"role": "system", "content": NOVA_PROMPT + "\n" + memory_text},
        *conversation_history,
        {"role": "user", "content": stt_text}
    ]

    # Nova brain
    nova_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={"model": GROQ_MODEL, "messages": messages},
        timeout=30
    )
    nova_response.raise_for_status()
    nova_reply = nova_response.json()["choices"][0]["message"]["content"]

    # Update conversation history
    conversation_history.append({"role": "user", "content": stt_text})
    conversation_history.append({"role": "assistant", "content": nova_reply})

    # ElevenLabs TTS
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": nova_reply,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}
    }

    audio_bytes = requests.post(tts_url, headers=headers, json=payload, timeout=60).content
    return Response(content=audio_bytes, media_type="audio/wav")


# -------------------------------------------------
# 2. Nova Text Endpoint (no audio)
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
        json={"model": GROQ_MODEL, "messages": messages}
    )

    ai_text = response.json()["choices"][0]["message"]["content"]

    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": ai_text})

    return {"response": ai_text}


# -------------------------------------------------
# 3. ElevenLabs Direct TTS Endpoint
# -------------------------------------------------
@app.post("/nova/speak")
async def nova_speak(request: Request):
    data = await request.json()
    text = data.get("text", "")

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}
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
