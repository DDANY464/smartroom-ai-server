FROM ollama/ollama:latest

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

# Copy your FastAPI app
COPY . .

# Start Ollama, pull model, then start FastAPI
CMD ["/bin/sh", "-c", "ollama serve & sleep 3 && ollama pull llama3.2:1b && uvicorn main:app --host 0.0.0.0 --port 8000"]
