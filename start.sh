#!/bin/sh

# Start Ollama server
ollama serve &

# Wait for Ollama to fully boot
sleep 8

# Pull model at runtime (safe)
ollama pull llama3.2:1b

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
