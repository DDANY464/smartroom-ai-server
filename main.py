from fastapi import FastAPI
import requests
import os

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.post("/nova")
async def nova_endpoint(payload: dict):
    user_text = payload.get("text", "")

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
            json=data
        )
        groq = response.json()
        model_reply = groq["choices"][0]["message"]["content"]

    except Exception as e:
        model_reply = f"Error: {str(e)}"

    return {"reply": model_reply}
