#!/bin/sh
ollama serve &
sleep 3
ollama pull llama3.2:1b
uvicorn main:app --host 0.0.0.0 --port 8000
