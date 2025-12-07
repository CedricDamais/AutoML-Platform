#!/usr/bin/env bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   AutoML Platform Launcher   ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed. Please install it first.${NC}"
    exit 1
fi

# All the ports used by the services
USED_PORTS=("5001" "8000" "8501")

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${BLUE}Cleanup...${NC}"
    for port in "${USED_PORTS[@]}"; do
        kill -9 $(lsof -t -i:$port -sTCP:LISTEN) 2> /dev/null
    done
}

cleanup

[ "$1" = '--stop' ] && exit

trap cleanup SIGINT SIGTERM

# Check if Redis is running (simple check)
if ! (echo > /dev/tcp/localhost/6379) >/dev/null 2>&1; then
    echo -e "${RED}Warning: Redis does not seem to be running on localhost:6379${NC}"
    echo -e "${RED}Please start Redis before running the platform.${NC}"
    echo -e "${BLUE}Tip: You can run 'docker run -d -p 6379:6379 redis'${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}[1/4] Starting MLflow Server...${NC}"
uv run mlflow server --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5001 --serve-artifacts &
MLFLOW_PID=$!

# Wait for MLflow to be ready
echo "Waiting for MLflow to start..."
for i in {1..30}; do
    if (echo > /dev/tcp/localhost/5001) >/dev/null 2>&1; then
        echo "MLflow is up!"
        break
    fi
    sleep 1
done

echo -e "${GREEN}[2/4] Starting API Server...${NC}"
uv run uvicorn src.api.main:app --reload --port 8000 &
API_PID=$!
sleep 2

echo -e "${GREEN}[3/4] Starting Job Worker...${NC}"
uv run python main.py &
WORKER_PID=$!

echo -e "${GREEN}[4/4] Starting Dashboard...${NC}"
uv run streamlit run src/dashboard/app.py --server.port 8501 &
DASHBOARD_PID=$!

sleep 2

clear

echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}Platform is running!${NC}"
echo -e "MLflow:    http://localhost:5001"
echo -e "API:       http://localhost:8000"
echo -e "Dashboard: http://localhost:8501"
echo -e "${BLUE}=========================================${NC}"
echo -e "Press Ctrl+C to stop all services."
