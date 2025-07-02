# Use official Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY backend ./backend
COPY preprocessing.py ./preprocessing.py
COPY run_backend_only.py ./run_backend_only.py

# Copy notebooks directory if it exists
COPY notebooks ./notebooks/

# Create necessary directories including project_info
RUN mkdir -p ./backend/uploads ./backend/preprocessed_unencoded ./backend/notebooks/project_info && \
    chmod -R 777 ./backend/uploads ./backend/preprocessed_unencoded ./backend/notebooks

# Copy ProjectID_Detail.xlsx to the correct location for preprocessing
RUN if [ -f ./notebooks/project_info/ProjectID_Detail.xlsx ]; then \
        cp ./notebooks/project_info/ProjectID_Detail.xlsx ./backend/notebooks/project_info/; \
    else \
        echo "Warning: ProjectID_Detail.xlsx not found"; \
    fi

# Set working directory to backend
WORKDIR /app/backend

# Expose backend port
EXPOSE 5000

# Set default command to run backend
CMD ["python", "app_workflow.py"]
