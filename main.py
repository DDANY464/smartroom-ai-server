from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()

HF_BASE_URL = os.getenv("HF_BASE_URL")
HF_MODEL = os.getenv("HF_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")

@app.get("/_debug/dns")
def dns_debug():
    import socket
    host = "api-inference.huggingface.co"
    try:
        addresses = socket.getaddrinfo(host, 443)
        resolved = True
        ips = list(set([a[4][0] for a in addresses]))
    except Exception as e:
        resolved = False
        ips = []
        return {"host": host, "resolved": resolved, "error": str(e)}

    return {"host": host, "resolved": resolved, "addresses": ips}

@app.post("/smartroom")
async def smartroom(request: Request):
    data = await request.json()

    motion = data.get("motion")
    noise = data.get("noise")
    command = data.get("command", "")

    payload = {
        "inputs": f"Motion: {motion}, Noise: {noise}, Command: {command}"
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    url = f"{HF_BASE_URL}/models/{HF_MODEL}"

    response = requests.post(url, headers=headers, json=payload)

    try:
        result = response.json()
    except:
        result = {"error": "Invalid JSON from Hugging Face", "raw": response.text}

    return {
        "input": data,
        "huggingface_response": result
    }
