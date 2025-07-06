# PSIREG DataFeed Deployment Guide

This guide provides comprehensive instructions for deploying the PSIREG Weather & Demand Data Pipeline in production environments.

## 🚀 Quick Start

1. **Install and Deploy**:
   ```bash
   cd deployment
   ./deploy.sh install
   ./deploy.sh deploy --detach
   ```

2. **Access Services**:
   - DataFeed API: http://localhost:8000
   - Grafana Dashboard: http://localhost:3000 (admin/admin)
   - Prometheus Metrics: http://localhost:9090
   - Redis Cache: localhost:6379

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+ with WSL2
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM (16GB recommended for production)
- **Storage**: 50GB+ available space
- **Network**: Internet access for data fetching

### Software Dependencies
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Python**: 3.8+ (for development)
- **Poetry**: Latest version (auto-installed)

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Weather APIs  │    │  Demand APIs    │    │   CSV/Parquet   │
│   (NREL, NOAA)  │    │ (ERCOT, CAISO)  │    │     Files       │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  WeatherDataExtractor   │
                    │   (Multi-source ETL)    │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     ETL Pipeline        │
                    │ (Transform & Validate)  │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Parquet Storage       │
                    │  (Compressed Data)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      DataFeed           │
                    │  (Streaming Engine)     │
                    └────────────┬────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                            │                            │
    ▼                            ▼                            ▼
┌─────────┐              ┌──────────────┐              ┌──────────┐
│  Redis  │              │ GridEngine   │              │ Grafana  │
│ Cache   │              │ Integration  │              │Dashboard │
└─────────┘              └──────────────┘              └──────────┘
```

## 🔧 Configuration

### Environment Variables

Create or edit `deployment/.env`:

```bash
# Environment
ENV=prod
LOG_LEVEL=info

# Weather API Configuration
WEATHER_API_KEY=your_api_key_here
WEATHER_API_TIMEOUT=60
WEATHER_API_RETRIES=5

# Performance Configuration
CACHE_SIZE_MB=500
CACHE_TTL_SECONDS=7200
MAX_WORKERS=8
BATCH_SIZE=2000
MEMORY_LIMIT_MB=4000

# Storage Configuration
DATA_PATH=/opt/psireg/data
PARQUET_STORAGE_PATH=/data/parquet
CSV_STORAGE_PATH=/data/csv

# Security
GRAFANA_PASSWORD=your_secure_password

# Data Validation
DATA_VALIDATION_ENABLED=true
COMPRESSION_LEVEL=9
```

### Performance Tuning

#### For High-Volume Production
```bash
# High-performance configuration
CACHE_SIZE_MB=2000
MAX_WORKERS=16
BATCH_SIZE=5000
MEMORY_LIMIT_MB=8000
COMPRESSION_LEVEL=6  # Lower compression for faster processing
```

#### For Resource-Constrained Environments
```bash
# Low-resource configuration
CACHE_SIZE_MB=100
MAX_WORKERS=2
BATCH_SIZE=500
MEMORY_LIMIT_MB=1000
COMPRESSION_LEVEL=9  # Maximum compression
```

## 📦 Deployment Methods

### Method 1: Automated Deployment (Recommended)

```bash
# Full deployment with monitoring
./deploy.sh deploy --env prod --data-path /opt/psireg/data --detach

# Check status
./deploy.sh status

# View logs
./deploy.sh logs
```

### Method 2: Manual Docker Compose

```bash
# Build images
docker-compose -f production.yml build

# Start services
docker-compose -f production.yml up -d

# Check health
docker-compose -f production.yml ps
```

### Method 3: Development Deployment

```bash
# Install dependencies
./deploy.sh install

# Run tests
./deploy.sh test

# Start in development mode
poetry run python -m psireg.sim.datafeed
```

## 🔍 Monitoring & Observability

### Grafana Dashboards

1. **DataFeed Performance Dashboard**:
   - Cache hit rates
   - Data processing throughput
   - API response times
   - Error rates

2. **System Resources Dashboard**:
   - CPU usage
   - Memory consumption
   - Disk I/O
   - Network traffic

3. **Grid Integration Dashboard**:
   - Asset conditions
   - Power flows
   - Frequency stability
   - Forecasting accuracy

### Prometheus Metrics

Key metrics exposed:
- `psireg_datafeed_requests_total`
- `psireg_datafeed_processing_duration_seconds`
- `psireg_cache_hits_total`
- `psireg_cache_misses_total`
- `psireg_data_points_processed_total`
- `psireg_forecast_accuracy_ratio`

### Health Checks

```bash
# Check all services
curl http://localhost:8000/health
curl http://localhost:3000/api/health
curl http://localhost:9090/-/healthy

# Or use the deployment script
./deploy.sh status
```

## 📊 Data Management

### Storage Structure

```
data/
├── parquet/
│   ├── weather/
│   │   ├── year=2024/
│   │   │   ├── month=01/
│   │   │   │   └── weather_20240101.parquet
│   │   │   └── month=02/
│   │   └── year=2023/
│   └── demand/
│       ├── year=2024/
│       └── year=2023/
├── csv/
│   ├── weather_raw.csv
│   └── demand_raw.csv
└── logs/
    ├── datafeed.log
    └── etl.log
```

### Data Retention

Configure data retention policies:

```bash
# Keep 90 days of data
find /data/parquet -name "*.parquet" -mtime +90 -delete

# Archive old data
tar -czf archive_$(date +%Y%m%d).tar.gz /data/parquet/year=2023/
```

## 🔐 Security Considerations

### API Keys Management

Store sensitive API keys securely:

```bash
# Use Docker secrets (recommended)
echo "your_api_key" | docker secret create weather_api_key -

# Or use environment variables
export WEATHER_API_KEY="your_api_key"
```

### Network Security

```bash
# Restrict access to internal services
# Only expose necessary ports externally
ports:
  - "127.0.0.1:8000:8000"  # DataFeed API
  - "127.0.0.1:3000:3000"  # Grafana (internal only)
```

### Data Encryption

Enable encryption for sensitive data:

```bash
# Encrypt Parquet files
DATA_ENCRYPTION_ENABLED=true
ENCRYPTION_KEY=your_encryption_key
```

## 🚨 Troubleshooting

### Common Issues

#### 1. High Memory Usage
```bash
# Check memory usage
docker stats

# Reduce cache size
CACHE_SIZE_MB=100

# Reduce batch size
BATCH_SIZE=500
```

#### 2. API Rate Limiting
```bash
# Increase retry attempts
WEATHER_API_RETRIES=10

# Add delays between requests
API_RATE_LIMIT_DELAY=1000  # milliseconds
```

#### 3. Data Quality Issues
```bash
# Enable verbose validation
DATA_VALIDATION_ENABLED=true
LOG_LEVEL=debug

# Check validation logs
./deploy.sh logs psireg-datafeed | grep VALIDATION
```

#### 4. Performance Issues
```bash
# Check performance metrics
curl http://localhost:8000/metrics

# Optimize workers
MAX_WORKERS=8

# Enable parallel processing
ENABLE_PARALLEL_PROCESSING=true
```

### Log Analysis

```bash
# View real-time logs
./deploy.sh logs -f

# Search for errors
./deploy.sh logs | grep ERROR

# Check specific service
./deploy.sh logs psireg-datafeed
./deploy.sh logs redis-cache
./deploy.sh logs monitoring
```

## 🔄 Backup & Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh - Backup script

BACKUP_DIR="/backup/psireg/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup Parquet data
tar -czf "$BACKUP_DIR/parquet_data.tar.gz" /data/parquet/

# Backup configuration
cp -r deployment/.env "$BACKUP_DIR/"
cp -r deployment/config/ "$BACKUP_DIR/"

# Backup Redis data
docker exec redis-cache redis-cli --rdb - > "$BACKUP_DIR/redis_dump.rdb"
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh - Restore script

BACKUP_DIR="/backup/psireg/$1"

# Stop services
./deploy.sh stop

# Restore data
tar -xzf "$BACKUP_DIR/parquet_data.tar.gz" -C /

# Restore configuration
cp "$BACKUP_DIR/.env" deployment/
cp -r "$BACKUP_DIR/config/" deployment/

# Restore Redis
docker exec redis-cache redis-cli --pipe < "$BACKUP_DIR/redis_dump.rdb"

# Start services
./deploy.sh start
```

## 🔄 Updates & Maintenance

### System Updates

```bash
# Update to latest version
git pull origin main

# Rebuild and redeploy
./deploy.sh build --force
./deploy.sh restart

# Run tests after update
./deploy.sh test
```

### Maintenance Tasks

```bash
# Weekly maintenance
./maintenance.sh weekly

# Monthly maintenance  
./maintenance.sh monthly

# Manual cleanup
./deploy.sh clean
```

## 📈 Performance Benchmarks

### Expected Performance

| Configuration | Throughput | Latency | Memory Usage |
|---------------|------------|---------|--------------|
| Small (2 CPU, 4GB) | 1K points/sec | <100ms | 2GB |
| Medium (4 CPU, 8GB) | 5K points/sec | <50ms | 4GB |
| Large (8 CPU, 16GB) | 15K points/sec | <20ms | 8GB |

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 🆘 Support & Documentation

### Getting Help

1. **Documentation**: Check the main README.md
2. **Integration Tests**: Run `./deploy.sh test`
3. **Logs**: Check `./deploy.sh logs` for detailed information
4. **Issues**: Report issues with detailed logs and configuration

### Useful Commands

```bash
# Quick health check
./deploy.sh status

# View performance metrics
curl http://localhost:8000/performance/stats

# Force restart all services
./deploy.sh restart

# Clean rebuild
./deploy.sh clean
./deploy.sh deploy --force

# Export logs for troubleshooting
./deploy.sh logs > psireg_logs_$(date +%Y%m%d).txt
```

---

For more detailed information, see the main project documentation and source code comments. 