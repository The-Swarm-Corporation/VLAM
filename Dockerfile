FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create a non-root user
RUN groupadd -r vlam && useradd -r -g vlam vlam
RUN chown -R vlam:vlam /app
USER vlam

# Expose port for potential web services
EXPOSE 8000

# Set default command
CMD ["python", "tests/test_vlam.py"]
