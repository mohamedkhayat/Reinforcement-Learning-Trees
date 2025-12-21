# ==============================================================================
# Dockerfile for RLT (Reinforcement Learning Trees) - LIGHTWEIGHT VERSION
# ==============================================================================
# Build:  docker build -t rlt-app .
# Run:    docker run -p 5000:5000 rlt-app
# Size:   ~300MB (optimized)
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install minimal Python dependencies for Flask app only
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    numpy==1.26.0 \
    pandas==2.1.0 \
    scikit-learn==1.3.0 \
    matplotlib==3.8.0 \
    joblib==1.3.0

# Copy ONLY what's needed for the Flask app
COPY src/RLT/ ./src/RLT/
COPY app/app.py ./app/
COPY app/templates/ ./app/templates/
COPY utils/dataset_wrapper.py ./utils/
COPY utils/helpers.py ./utils/
COPY scripts/data_preparation.py ./scripts/
COPY datasets/ ./datasets/

# Create __init__.py files for imports
RUN touch ./src/__init__.py && \
    touch ./utils/__init__.py && \
    touch ./scripts/__init__.py

# Add src to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Expose the Flask port
EXPOSE 5000

# Run with gunicorn (lighter config)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300", "app.app:app"]
