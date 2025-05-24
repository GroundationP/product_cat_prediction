# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libjpeg-dev \
    zlib1g-dev \
    procps \
    && rm -rf /var/lib/apt/lists/*


# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app
COPY ./app .
COPY run_server_docker.sh .
COPY run_server_docker.sh ./app

# Make script executable
RUN chmod +x run_server_docker.sh

# Expose the port
EXPOSE 8000
EXPOSE 5001

# Run the FastAPI app
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Start both services
CMD ["./run_server_docker.sh"]

