from fastapi import FastAPI
import requests
import os

app = FastAPI()

HF_API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

PIPELINE_URL = "https://api-inference.huggingface.co/pipeline/text-generation"

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

    data = {
        "model": MODEL_NAME,
        "inputs": user_text,
        "parameters": {
            "max_new_tokens": 50
        }
    }

    try:
        response = requests.post(PIPELINE_URL, headers=headers, json=data)
        hf = response.json()

        model_reply = hf.get("generated_text", "No reply")

    except Exception as e:
        model_reply = f"Error: {str(e)}"

    return {"reply": model_reply}
