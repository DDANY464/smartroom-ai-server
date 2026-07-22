import os
import logging
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova-debug-backend")

# -------------------------------------------------
# FastAPI App
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
# Environment Variables
# -------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger.info("--------------------------------------------------")
logger.info("DEBUG: Starting Nova backend")
logger.info(f"DEBUG: GROQ_API_KEY loaded: {GROQ_API_KEY}")
logger.info("--------------------------------------------------")

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/")
def home():
    return {"status": "ok", "message": "Nova backend running (debug mode)"}


@app.post("/nova")
async def nova_endpoint(payload: dict):
    """
    Debug endpoint that prints EVERYTHING sent to Groq.
    """

    logger.info("--------------------------------------------------")
    logger.info("DEBUG: /nova endpoint hit")
    logger.info(f"DEBUG: Incoming payload: {payload}")

    if GROQ_API_KEY is None:
        logger.error("DEBUG: GROQ_API_KEY is missing!")
        return {"reply": "Error: GROQ_API_KEY is not set in environment variables"}

    user_text = payload.get("text", "")
    if not user_text:
        logger.error("DEBUG: Missing 'text' field")
        return {"reply": "Error: 'text' field is missing or empty in JSON body"}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": user_text}
        ]
    }

    logger.info("DEBUG: Sending request to Groq...")
    logger.info(f"DEBUG: URL: {url}")
    logger.info(f"DEBUG: Headers: {headers}")
    logger.info(f"DEBUG: JSON Body: {data}")

    try:
        response = requests.post(url, headers=headers, json=data)

        logger.info("DEBUG: Groq responded!")
        logger.info(f"DEBUG: Status Code: {response.status_code}")
        logger.info(f"DEBUG: Raw Response Text: {response.text}")

        response.raise_for_status()

        groq_json = response.json()
        reply = groq_json["choices"][0]["message"]["content"]

        logger.info(f"DEBUG: Parsed reply: {reply}")
        logger.info("--------------------------------------------------")

        return {"reply": reply}

    except Exception as e:
        logger.error("DEBUG: Groq API ERROR!")
        logger.error(f"DEBUG: Exception: {str(e)}")
        logger.info("--------------------------------------------------")
        return {"reply": f"Error: {str(e)}"}


# -------------------------------------------------
# Railway Port Binding
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    logger.info(f"DEBUG: Starting Uvicorn on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
