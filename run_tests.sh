#!/usr/bin/env bash
# Test runner script for AutoML Platform (using uv)

set -e

echo "================================"
echo "AutoML Platform Test Suite"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${2}${1}${NC}"
}

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_status "uv not found. Please install uv first." "$RED"
    exit 1
fi

print_status "Running all tests..." "$GREEN"
echo ""

# Run tests based on argument
case "${1:-all}" in
    "unit")
        print_status "Running unit tests only..." "$GREEN"
        uv run pytest tests/test_api_routes.py tests/test_job_scheduler.py -v
        ;;
    "integration")
        print_status "Running integration tests only..." "$GREEN"
        uv run pytest tests/test_redis_workflow.py -v
        ;;
    "coverage")
        print_status "Running tests with coverage report..." "$GREEN"
        uv run pytest --cov=src --cov-report=html --cov-report=term-missing
        print_status "Coverage report generated in htmlcov/index.html" "$YELLOW"
        ;;
    "fast")
        print_status "Running fast tests only (excluding slow tests)..." "$GREEN"
        uv run pytest -m "not slow" -v
        ;;
    "all"|*)
        print_status "Running all tests..." "$GREEN"
        uv run pytest -v
        ;;
esac

echo ""
print_status "Tests completed!" "$GREEN"
