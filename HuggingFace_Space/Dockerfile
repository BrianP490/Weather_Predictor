# Use an official Python runtime as a parent image.
FROM python:3.10-slim-bullseye

# Set the working directory in the container.
WORKDIR /app

# Install system dependencies required by PyTorch and others.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements.txt file into the container and install dependencies.
COPY requirements.txt .

# Update pip and setuptools before installing other dependencies to prevent errors
RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire application code, including the model weights, into the container.
# This assumes your model is at ./app/model_weights/model_policy.pth relative to the project root.
COPY . /app

# Expose the port that Streamlit runs on.
EXPOSE 8501

# Command to run the Streamlit application when the container starts.
ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]