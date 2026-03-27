FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System deps (curl needed by entrypoint health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY --chown=appuser requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy application source
COPY --chown=appuser project/ ./project/
COPY --chown=appuser docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Pre-create volume mount points as appuser so Docker initialises the
# named volumes with the correct ownership on first mount.
RUN mkdir -p /app/project/parent_store && chown -R appuser:appuser /app/project/parent_store

USER appuser

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
