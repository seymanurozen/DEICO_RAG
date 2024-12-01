# Use an official Python base image with Ubuntu
FROM python:3.9-slim-buster

# Install necessary system packages
RUN apt-get update && \
    apt-get install -y curl unzip build-essential git

# Install Ollama
RUN curl -O https://ollama.ai/install.sh && \
    bash install.sh

# Set the working directory
WORKDIR /app

# Copy your application code to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for Streamlit and Ollama
EXPOSE 8501

# Start both Ollama and the Streamlit app
CMD ["bash", "-c", "ollama serve & streamlit run your_streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]