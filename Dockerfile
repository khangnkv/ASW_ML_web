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

# Create the EXACT directory structure expected by preprocessing.py
RUN mkdir -p /app/backend/uploads /app/backend/preprocessed_unencoded /app/backend/notebooks/project_info

# CRITICAL: Copy the ProjectID_Detail.xlsx to the EXACT location expected
COPY backend/notebooks/project_info/ProjectID_Detail.xlsx /app/backend/notebooks/project_info/ProjectID_Detail.xlsx

# Verify the file is in the expected location
RUN echo "Verifying ProjectID_Detail.xlsx in expected location:" && \
    ls -la /app/backend/notebooks/project_info/ProjectID_Detail.xlsx && \
    echo "File size: $(du -h /app/backend/notebooks/project_info/ProjectID_Detail.xlsx)"

# Now copy all other files
COPY backend ./backend
COPY preprocessing.py ./preprocessing.py
COPY run_backend_only.py ./run_backend_only.py

# Copy notebooks directory if it exists
COPY notebooks ./notebooks/ 2>/dev/null || echo "No notebooks directory found"

# Set proper permissions
RUN chmod -R 777 /app/backend/uploads /app/backend/preprocessed_unencoded /app/backend/notebooks

# Final verification - ensure the file is still there after all copies
RUN echo "=== FINAL VERIFICATION ===" && \
    ls -la /app/backend/notebooks/project_info/ && \
    if [ -f "/app/backend/notebooks/project_info/ProjectID_Detail.xlsx" ]; then \
        echo "SUCCESS: ProjectID_Detail.xlsx ready at expected location"; \
        echo "File size: $(du -h /app/backend/notebooks/project_info/ProjectID_Detail.xlsx)"; \
    else \
        echo "ERROR: ProjectID_Detail.xlsx missing from expected location"; \
        exit 1; \
    fi

# Set working directory to backend
WORKDIR /app/backend

# Expose backend port
EXPOSE 5000

# Set default command to run backend
CMD ["python", "app_workflow.py"]
