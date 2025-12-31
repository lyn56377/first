FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY . /app

# Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports
EXPOSE 8000
EXPOSE 11434

# Start Ollama + API
CMD ollama serve & \
    sleep 5 && \
    ollama pull phi3:mini && \
    uvicorn main:app --host 0.0.0.0 --port 8000
