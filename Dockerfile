# syntax=docker/dockerfile:1

# Build stage - install dependencies with build tools
FROM python:3.12-slim-bookworm AS builder

# Install system dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    proj-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies to a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Production stage - minimal runtime image
FROM python:3.12-slim-bookworm AS production

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal32 \
    libproj25 \
    proj-bin \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/

# Create temp directory for IFC file generation (writable by appuser)
RUN mkdir -p /app/tmp && chown appuser:appgroup /app/tmp
ENV TMPDIR=/app/tmp
ENV PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Use PORT env variable (Sevalla sets this)
ENV PORT=8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.getenv('PORT', '8080'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# Run with uvicorn
CMD ["sh", "-c", "uvicorn src.rest_api:app --host 0.0.0.0 --port ${PORT}"]

