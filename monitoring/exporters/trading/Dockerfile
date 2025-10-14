FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    aiosqlite==0.19.0

# Copy exporter script
COPY trading_exporter.py .

# Create user for security
RUN useradd -r -u 1000 exporter && \
    chown -R exporter:exporter /app

USER exporter

# Expose metrics port
EXPOSE 8000

# Run exporter
CMD ["python", "-u", "trading_exporter.py"]
