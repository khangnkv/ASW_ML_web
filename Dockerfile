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
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy backend and model files, but exclude uploads and preprocessed_unencoded
COPY backend ./backend
COPY backend/model ./backend/model
RUN mkdir -p ./backend/notebooks/project_info
COPY backend/notebooks/project_info/ProjectID_Detail.xlsx ./backend/notebooks/project_info/ProjectID_Detail.xlsx
COPY preprocessing.py ./preprocessing.py
COPY run_backend_only.py ./run_backend_only.py

# Ensure upload directories exist and are writable
RUN mkdir -p ./backend/uploads ./backend/preprocessed_unencoded \
    && chmod -R 777 ./backend/uploads ./backend/preprocessed_unencoded

# Expose backend port
EXPOSE 5000

# Set default command to run backend
CMD ["python", "backend/app_workflow.py"]
