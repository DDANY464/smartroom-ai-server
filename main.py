import os
import logging
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova-backend")

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI()

# Allow all CORS (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Environment Variables
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    logger.error("GROQ_API_KEY is missing! Set it in Railway environment variables.")

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/")
def home():
    return {"status": "ok", "message": "Nova backend running"}


@app.post("/nova")
async def nova_endpoint(payload: dict):
    """
    Accepts: { "text": "hello nova" }
    Returns: { "reply": "..." }
    """

    if GROQ_API_KEY is None:
        return {"reply": "Error: GROQ_API_KEY is not set in environment variables"}

    user_text = payload.get("text", "")
    if not user_text:
        return {"reply": "Error: 'text' field is missing or empty in JSON body"}

    logger.info(f"Nova request received: {user_text}")

    url = "https://api.groq.com/openai/v1/chat/completions"
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
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        groq_json = response.json()
        reply = groq_json["choices"][0]["message"]["content"]

        logger.info(f"Nova reply: {reply}")
        return {"reply": reply}

    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        return {"reply": f"Error: {str(e)}"}


# -------------------------------------------------
# Railway Port Binding
# -------------------------------------------------
# Railway injects PORT automatically
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
