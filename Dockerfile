FROM ollama/ollama:latest

# Remove Ollama's ENTRYPOINT so our script runs normally
ENTRYPOINT []

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

# Copy the rest of your app
COPY . .

# Copy and enable the startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Run Ollama + pull model + start FastAPI
CMD ["/bin/sh", "/start.sh"]
