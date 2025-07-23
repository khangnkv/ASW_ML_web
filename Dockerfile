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

# Create necessary directories first
RUN mkdir -p /app/backend/uploads /app/backend/preprocessed_unencoded /app/backend/notebooks/project_info /app/notebooks/project_info

# CRITICAL: Copy ProjectID_Detail.xlsx FIRST before copying other files
# This ensures the file is definitely there
COPY backend/notebooks/project_info/ProjectID_Detail.xlsx /app/backend/notebooks/project_info/ProjectID_Detail.xlsx

# Verify the file was copied correctly
RUN ls -la /app/backend/notebooks/project_info/ProjectID_Detail.xlsx && \
    echo "ProjectID_Detail.xlsx successfully copied to /app/backend/notebooks/project_info/"

# Now copy other files
COPY backend ./backend
COPY preprocessing.py ./preprocessing.py
COPY run_backend_only.py ./run_backend_only.py
COPY notebooks ./notebooks/

# Create additional copies in fallback locations
RUN cp /app/backend/notebooks/project_info/ProjectID_Detail.xlsx /app/notebooks/project_info/ProjectID_Detail.xlsx && \
    mkdir -p /app/project_info && \
    cp /app/backend/notebooks/project_info/ProjectID_Detail.xlsx /app/project_info/ProjectID_Detail.xlsx

# Set proper permissions
RUN chmod -R 777 /app/backend/uploads /app/backend/preprocessed_unencoded /app/backend/notebooks /app/notebooks

# Final verification
RUN echo "=== FINAL VERIFICATION ===" && \
    ls -la /app/backend/notebooks/project_info/ && \
    ls -la /app/notebooks/project_info/ && \
    ls -la /app/project_info/ && \
    echo "=== END VERIFICATION ==="

# Set working directory to backend
WORKDIR /app/backend

# Expose backend port
EXPOSE 5000

# Set default command to run backend
CMD ["python", "app_workflow.py"]
