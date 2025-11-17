FROM python:3.11-slim

# Install system packages needed for sklearn/xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY model.bin dv.bin serve.py ./

# Expose API port
EXPOSE 8000

# Run server
CMD ["uvicorn", "serve:app", "--host=0.0.0.0", "--port=8000"]
