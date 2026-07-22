# main.py
import os
import logging
from typing import Dict

import httpx               # async HTTP client (replaces requests)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------------------------------
# Logging – never print the raw secret
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("nova-debug-backend")

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="Nova Groq Proxy",
    description="Thin wrapper that forwards a user message to a Groq model.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod (e.g. ["https://my‑frontend.com"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Environment variables
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("❗️ GROQ_API_KEY is not set in the environment")

# Primary model – you can override via env var on Railway
GROQ_MODEL = os.getenv("GROQ_MODEL", "phi-3-mini-4k-instruct")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

log.info("🚀 Nova backend starting – model=%s", GROQ_MODEL)

# -------------------------------------------------
# Pydantic request schema (validation + docs)
# -------------------------------------------------
class NovaPayload(BaseModel):
    text: str = Field(..., min_length=1, description="User‑provided prompt")

# -------------------------------------------------
# Shared async HTTP client (connection pool)
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

    log.debug("POST %s – model=%s – user_text=%s", GROQ_ENDPOINT, model, user_text[:30] + "...")

    resp = await client.post(GROQ_ENDPOINT, json=payload, headers=headers)

    if resp.is_success:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    else:
        # Try to surface a useful message from Groq
        try:
            err = resp.json()
            msg = err.get("error", {}).get("message", resp.text)
        except Exception:
            msg = resp.text
        raise HTTPException(status_code=resp.status_code, detail=f"Groq error: {msg}")


# -------------------------------------------------
# Health‑check route
# -------------------------------------------------
@app.get("/", summary="Simple health check")
async def root():
    return {"status": "ok", "message": "Nova backend running (debug mode)"}


# -------------------------------------------------
# Main endpoint
# -------------------------------------------------
@app.post("/nova", summary="Forward a user prompt to Groq")
async def nova_endpoint(payload: NovaPayload):
    log.info("🔔 /nova hit – payload=%s", payload.dict())

    try:
        reply = await _call_groq(payload.text, GROQ_MODEL)
        log.info("✅ Groq reply received (len=%d)", len(reply))
        return {"reply": reply}
    except HTTPException as exc:
        # If the model has been de‑commissioned, give a clear hint
        if exc.status_code == 400 and "decommissioned" in str(exc.detail).lower():
            log.warning("Model %s is decommissioned – ask Groq for a replacement", GROQ_MODEL)
        raise  # FastAPI will turn this into a proper JSON error response


# -------------------------------------------------
# Railway / local entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8080))
    log.info("🚀 Starting uvicorn on 0.0.0.0:%s", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
