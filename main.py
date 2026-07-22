# main.py
import os
import logging
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("nova-backend")

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="Nova Groq Proxy",
    description="Thin wrapper that forwards a user message to a Groq model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Environment variables
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❗ GROQ_API_KEY is not set in the environment")

# Correct fallback model (Groq‑supported)
GROQ_MODEL = os.getenv("GROQ_MODEL", "phi-3-mini-4k-instruct-q4")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

log.info("🚀 Nova backend starting – model=%s", GROQ_MODEL)

# -------------------------------------------------
# Request schema
# -------------------------------------------------
class NovaPayload(BaseModel):
    text: str = Field(..., min_length=1, description="User prompt")

# -------------------------------------------------
# Shared async HTTP client
# -------------------------------------------------
client = httpx.AsyncClient(timeout=30.0)

async def _call_groq(user_text: str, model: str) -> str:
    """Send the chat request to Groq and return the assistant reply."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    log.debug("POST %s – model=%s – text=%s", GROQ_ENDPOINT, model, user_text[:40])

    resp = await client.post(GROQ_ENDPOINT, json=payload, headers=headers)

    if resp.is_success:
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # Extract Groq error message
    try:
        err = resp.json()
        msg = err.get("error", {}).get("message", resp.text)
    except Exception:
        msg = resp.text

    raise HTTPException(status_code=resp.status_code, detail=f"Groq error: {msg}")

# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Nova backend running"}

# -------------------------------------------------
# Main endpoint
# -------------------------------------------------
@app.post("/nova")
async def nova_endpoint(payload: NovaPayload):
    log.info("🔔 /nova hit – payload=%s", payload.dict())

    try:
        reply = await _call_groq(payload.text, GROQ_MODEL)
        log.info("✅ Groq reply received (len=%d)", len(reply))
        return {"reply": reply}

    except HTTPException as exc:
        if exc.status_code == 400 and "decommissioned" in str(exc.detail).lower():
            log.warning("⚠ Model %s is decommissioned – update GROQ_MODEL", GROQ_MODEL)
        raise

# -------------------------------------------------
# Local / Railway entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    log.info("🚀 Starting uvicorn on 0.0.0.0:%s", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port)
