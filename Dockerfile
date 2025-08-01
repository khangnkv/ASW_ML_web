# Filename: Dockerfile (place in your project root)

# Stage 1: Base Image
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for the application and health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgl1-mesa-glx \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code into the container
COPY backend ./backend
COPY preprocessing.py ./preprocessing.py
COPY run_backend_only.py ./run_backend_only.py
COPY backend/notebooks ./backend/notebooks/

# Create directories for the application, including mount points for volumes.
# This ensures the container can run even without docker-compose.
# The volume mounts in docker-compose will overlay these directories.
RUN mkdir -p \
    ./backend/uploads \
    ./backend/preprocessed_unencoded \
    ./backend/model \
    ./data \
    ./logs && \
    chmod -R 755 ./backend/uploads ./backend/preprocessed_unencoded ./backend/notebooks ./data ./logs

# Add the healthcheck instruction to the image metadata
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# Create a startup script to run the application
RUN echo '#!/bin/sh' > /app/start.sh && \
    echo 'echo "Starting ML Prediction System Backend..."' >> /app/start.sh && \
    echo 'python app_workflow.py' >> /app/start.sh && \
    chmod +x /app/start.sh

# Set the working directory to the backend folder where the app runs
WORKDIR /app/backend

# Expose the port the app runs on
EXPOSE 5000

# Set the default command to run the startup script
CMD ["/app/start.sh"]