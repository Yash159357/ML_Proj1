# =========================
# Stage 1: Builder
# =========================
FROM python:3.8-slim AS builder

# Set working directory inside container
WORKDIR /app

# Copy only runtime dependencies first for caching
COPY runtime.txt ./requirements.txt

# Build wheels for all dependencies
RUN pip wheel --wheel-dir=/wheels -r requirements.txt

# =========================
# Stage 2: Final image
# =========================
FROM python:3.8-slim

WORKDIR /app

# Copy pre-built wheels from builder
COPY --from=builder /wheels /wheels

# Copy application code and assets
COPY app.py ./
COPY src/ ./src
COPY templates/ ./templates
COPY static/ ./static
COPY artifact/model.pkl ./artifact/model.pkl
COPY artifact/preprocessor.pkl ./artifact/preprocessor.pkl

# Install all wheels (no cache)
RUN pip install --no-cache-dir /wheels/*

# Expose port to host
EXPOSE 8080

# Default environment variable
ENV PORT=8080

# Start Gunicorn server
CMD ["gunicorn", "app:application", "--bind", "0.0.0.0:8080"]
