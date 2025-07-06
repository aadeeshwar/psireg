#!/bin/bash

# PSIREG DataFeed Deployment Script
# This script sets up and deploys the PSIREG Weather & Demand Data Pipeline

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$SCRIPT_DIR/production.yml"
ENV_FILE="$SCRIPT_DIR/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
PSIREG DataFeed Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    install     Install dependencies and set up environment
    build       Build Docker images
    deploy      Deploy the system using Docker Compose
    start       Start the deployed system
    stop        Stop the running system
    restart     Restart the system
    status      Check system status
    logs        Show system logs
    test        Run integration tests
    clean       Clean up Docker resources
    help        Show this help message

Options:
    --env [dev|prod]    Set environment (default: prod)
    --data-path PATH    Set data storage path (default: ./data)
    --force             Force rebuild without cache
    --detach            Run in detached mode
    --verbose           Enable verbose output

Examples:
    $0 install
    $0 deploy --env prod --data-path /opt/psireg/data
    $0 start --detach
    $0 logs
    $0 test

Environment Variables:
    WEATHER_API_KEY     API key for weather data services
    GRAFANA_PASSWORD    Password for Grafana dashboard
    DATA_PATH           Path for data storage
    LOG_LEVEL           Logging level (debug, info, warning, error)
EOF
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    # Check Poetry
    if ! command -v poetry &> /dev/null; then
        log_warning "Poetry is not installed. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
    fi
    
    log_success "Prerequisites check completed"
}

# Create environment file
create_env_file() {
    log_info "Creating environment file..."
    
    if [[ ! -f "$ENV_FILE" ]]; then
        cat > "$ENV_FILE" << EOF
# PSIREG DataFeed Environment Configuration
ENV=${ENVIRONMENT:-prod}
LOG_LEVEL=${LOG_LEVEL:-info}

# Weather API Configuration
WEATHER_API_KEY=${WEATHER_API_KEY:-your_api_key_here}
WEATHER_API_TIMEOUT=60
WEATHER_API_RETRIES=5

# Performance Configuration
CACHE_SIZE_MB=500
CACHE_TTL_SECONDS=7200
MAX_WORKERS=8
BATCH_SIZE=2000
MEMORY_LIMIT_MB=4000

# Storage Configuration
DATA_PATH=${DATA_PATH:-./data}
PARQUET_STORAGE_PATH=/data/parquet
CSV_STORAGE_PATH=/data/csv

# Security Configuration
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-admin}

# Data Validation
DATA_VALIDATION_ENABLED=true
COMPRESSION_LEVEL=9
EOF
        log_success "Environment file created at $ENV_FILE"
        log_warning "Please edit $ENV_FILE to configure your settings"
    else
        log_info "Environment file already exists at $ENV_FILE"
    fi
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    cd "$PROJECT_ROOT"
    
    # Install Python dependencies
    poetry install
    
    # Create data directories
    mkdir -p "${DATA_PATH:-./data}/parquet"
    mkdir -p "${DATA_PATH:-./data}/csv"
    mkdir -p logs
    
    log_success "Dependencies installed successfully"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    BUILD_ARGS=""
    if [[ "$FORCE_BUILD" == "true" ]]; then
        BUILD_ARGS="--no-cache"
    fi
    
    docker-compose -f "$COMPOSE_FILE" build $BUILD_ARGS
    
    log_success "Docker images built successfully"
}

# Deploy system
deploy_system() {
    log_info "Deploying PSIREG DataFeed system..."
    
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        create_env_file
    fi
    
    cd "$SCRIPT_DIR"
    
    # Deploy with Docker Compose
    DEPLOY_ARGS="--env-file $ENV_FILE"
    if [[ "$DETACH" == "true" ]]; then
        DEPLOY_ARGS="$DEPLOY_ARGS -d"
    fi
    
    docker-compose -f "$COMPOSE_FILE" up $DEPLOY_ARGS
    
    log_success "PSIREG DataFeed system deployed successfully"
}

# Start system
start_system() {
    log_info "Starting PSIREG DataFeed system..."
    
    cd "$SCRIPT_DIR"
    
    START_ARGS=""
    if [[ "$DETACH" == "true" ]]; then
        START_ARGS="-d"
    fi
    
    docker-compose -f "$COMPOSE_FILE" start $START_ARGS
    
    log_success "PSIREG DataFeed system started"
    
    # Show service URLs
    echo ""
    log_info "Service URLs:"
    echo "  DataFeed API: http://localhost:8000"
    echo "  Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "  Prometheus Metrics: http://localhost:9090"
    echo "  Redis Cache: localhost:6379"
}

# Stop system
stop_system() {
    log_info "Stopping PSIREG DataFeed system..."
    
    cd "$SCRIPT_DIR"
    docker-compose -f "$COMPOSE_FILE" stop
    
    log_success "PSIREG DataFeed system stopped"
}

# Restart system
restart_system() {
    log_info "Restarting PSIREG DataFeed system..."
    
    stop_system
    start_system
    
    log_success "PSIREG DataFeed system restarted"
}

# Check system status
check_status() {
    log_info "Checking system status..."
    
    cd "$SCRIPT_DIR"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    log_info "Health checks:"
    
    # Check DataFeed health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "DataFeed API: Healthy"
    else
        log_error "DataFeed API: Unhealthy"
    fi
    
    # Check Grafana
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log_success "Grafana: Healthy"
    else
        log_error "Grafana: Unhealthy"
    fi
    
    # Check Prometheus
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log_success "Prometheus: Healthy"
    else
        log_error "Prometheus: Unhealthy"
    fi
}

# Show logs
show_logs() {
    log_info "Showing system logs..."
    
    cd "$SCRIPT_DIR"
    
    if [[ -n "$1" ]]; then
        # Show logs for specific service
        docker-compose -f "$COMPOSE_FILE" logs -f "$1"
    else
        # Show logs for all services
        docker-compose -f "$COMPOSE_FILE" logs -f
    fi
}

# Run tests
run_tests() {
    log_info "Running integration tests..."
    
    cd "$PROJECT_ROOT"
    
    # Run the integration tests
    poetry run python -m pytest tests/test_datafeed_integration.py -v
    
    if [[ $? -eq 0 ]]; then
        log_success "All integration tests passed"
    else
        log_error "Some integration tests failed"
        exit 1
    fi
}

# Clean up
cleanup() {
    log_info "Cleaning up Docker resources..."
    
    cd "$SCRIPT_DIR"
    
    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down
    
    # Remove images if requested
    if [[ "$1" == "--all" ]]; then
        log_warning "Removing all PSIREG images..."
        docker images | grep psireg | awk '{print $3}' | xargs -r docker rmi
        
        # Remove volumes
        log_warning "Removing all volumes..."
        docker-compose -f "$COMPOSE_FILE" down -v
    fi
    
    log_success "Cleanup completed"
}

# Parse command line arguments
COMMAND=""
ENVIRONMENT="prod"
DATA_PATH="./data"
FORCE_BUILD="false"
DETACH="false"
VERBOSE="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        install|build|deploy|start|stop|restart|status|logs|test|clean|help)
            COMMAND="$1"
            shift
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --force)
            FORCE_BUILD="true"
            shift
            ;;
        --detach)
            DETACH="true"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            set -x
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_info "PSIREG DataFeed Deployment Script"
    log_info "Environment: $ENVIRONMENT"
    log_info "Data Path: $DATA_PATH"
    echo ""
    
    case $COMMAND in
        install)
            check_prerequisites
            create_env_file
            install_dependencies
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        deploy)
            check_prerequisites
            create_env_file
            build_images
            deploy_system
            ;;
        start)
            start_system
            ;;
        stop)
            stop_system
            ;;
        restart)
            restart_system
            ;;
        status)
            check_status
            ;;
        logs)
            show_logs "$2"
            ;;
        test)
            run_tests
            ;;
        clean)
            cleanup "$2"
            ;;
        help|"")
            show_help
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Export environment variables
export ENVIRONMENT
export DATA_PATH

# Run main function
main "$@" 