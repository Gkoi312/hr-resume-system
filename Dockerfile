FROM python:3.10-slim

WORKDIR /app

# Install system deps for pymupdf / opencv / Paddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install demo deps
RUN pip install --no-cache-dir streamlit pandas

# Copy app code
COPY . .

# Expose API port
EXPOSE 8000

# Default: run API server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
