# Use Python 3.12 as base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --only=main --no-interaction --no-ansi

# Create non-root user
RUN useradd --create-home --shell /bin/bash psireg

# Create necessary directories
RUN mkdir -p /app/logs /data/parquet /data/csv && \
    chown -R psireg:psireg /app /data

# Copy application code
COPY --chown=psireg:psireg src/ /app/src/
COPY --chown=psireg:psireg tests/ /app/tests/
COPY --chown=psireg:psireg examples/ /app/examples/
COPY --chown=psireg:psireg README.md /app/

# Add src directory to Python path
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Switch to non-root user
USER psireg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import psireg.sim.datafeed; print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "psireg.sim.datafeed"] 