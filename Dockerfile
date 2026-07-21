FROM ollama/ollama:latest

SHELL ["/bin/sh", "-c"]

EXPOSE 8000

RUN apt-get update && apt-get install -y python3 python3-pip

# Pull model during build (safe)
RUN ollama pull llama3.2:1b

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --break-system-packages -r requirements.txt

COPY . .

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/bin/sh", "/start.sh"]
