from fastapi import FastAPI
import requests
import os

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_URL = os.getenv("MODEL_URL")

@app.post("/nova")
async def nova_endpoint(payload: dict):
    user_text = payload.get("text", "")

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {"inputs": user_text}

    response = requests.post(MODEL_URL, headers=headers, json=data)

    try:
        model_reply = response.json()[0]["generated_text"]
    except Exception:
        model_reply = "Error: Invalid model response."

    return {"reply": model_reply}
