from fastapi import FastAPI
import requests
import os

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_URL = os.getenv("MODEL_URL")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/nova")
async def nova_endpoint(payload: dict):
    user_text = payload.get("text", "")

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {"inputs": user_text}

    try:
        response = requests.post(MODEL_URL, headers=headers, json=data)
        hf = response.json()

        if isinstance(hf, list):
            model_reply = (
                hf[0].get("generated_text")
                or hf[0].get("text")
                or "No reply"
            )
        else:
            model_reply = (
                hf.get("generated_text")
                or hf.get("text")
                or "No reply"
            )

    except Exception as e:
        model_reply = f"Error: {str(e)}"

    return {"reply": model_reply}
