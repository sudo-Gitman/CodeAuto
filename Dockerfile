# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for GUI and other packages
RUN apt-get update && apt-get install -y \
    python3-tk \
    libx11-6 \
    libxext-dev \
    libxrender1 \
    libxinerama1 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxfixes3 \
    libxcomposite1 \
    libxdamage1 \
    libxkbfile1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for persistent data
RUN mkdir -p /app/generated_code /app/logs /app/plugins

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Command to run the application
CMD ["python", "gui.py"]
