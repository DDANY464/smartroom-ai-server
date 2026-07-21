FROM ollama/ollama:latest

# Override Ollama's entrypoint completely
ENTRYPOINT ["/bin/sh", "/start.sh"]

# Install Python
RUN apt-get update && apt-get install -y python3 python3-pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

# Copy your app
COPY . .

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh
