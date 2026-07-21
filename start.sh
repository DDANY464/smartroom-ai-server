#!/bin/sh

ollama serve &

sleep 5

uvicorn main:app --host 0.0.0.0 --port 8000
