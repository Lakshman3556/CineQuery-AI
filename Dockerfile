FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for compiling any packages if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install them
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application files
COPY backend ./backend

# Hugging Face Spaces expects the app to run on port 7860 by default
EXPOSE 7860

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
