# Use an official Python base image with Ubuntu
FROM python:3.11-slim

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y curl unzip build-essential git

# Install Ollama
RUN curl -O https://ollama.ai/install.sh && \
    bash install.sh

RUN ollama serve & sleep 5 && ollama pull llama3.2:1b
RUN ollama serve & sleep 5 && ollama pull llama3.2:3b
RUN ollama serve & sleep 5 && ollama pull llama3.2:latest
RUN ollama serve & sleep 5 && ollama pull nomic-embed-text
RUN ollama serve & sleep 5 && ollama pull mxbai-embed-large

# Set the working directory
WORKDIR /app

# Copy your application code to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for Streamlit and Ollama
EXPOSE 8080

# Start both Ollama and the Streamlit app
CMD ["bash", "-c", "ollama serve & streamlit run app.py --server.port=8080 --server.address=0.0.0.0"]