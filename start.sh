#!/bin/sh

# Start Ollama server in background
ollama serve &

# Wait for Ollama to boot
sleep 5

# Pull your model
ollama pull llama3.2:1b

# Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000
