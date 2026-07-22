from fastapi import FastAPI
import requests
import os

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@app.get("/")
def home():
    return {"status": "ok", "message": "Nova backend running"}


@app.post("/nova")
async def nova_endpoint(payload: dict):
    if GROQ_API_KEY is None:
        return {"reply": "Error: GROQ_API_KEY is not set in environment variables"}

    user_text = payload.get("text", "")
    if not user_text:
        return {"reply": "Error: 'text' field is missing or empty in JSON body"}

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "user", "content": user_text}
        ]
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=20
        )
        response.raise_for_status()
        groq = response.json()
        model_reply = groq["choices"][0]["message"]["content"]

    except Exception as e:
        model_reply = f"Error: {str(e)}"

    return {"reply": model_reply}
