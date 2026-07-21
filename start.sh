#!/bin/sh

# Start Ollama server in background
ollama serve &

# Wait for Ollama to boot
sleep 5

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
