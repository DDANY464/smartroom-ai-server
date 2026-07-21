FROM ollama/ollama:latest

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy your FastAPI app
COPY . .

# Pull the local AI model
RUN ollama pull llama3.2:1b

# Start Ollama + your FastAPI server
CMD ollama serve & uvicorn main:app --host 0.0.0.0 --port 8000
