#!/usr/bin/env bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

FRONTEND_DIR="src/dashboard"
FRONTEND_PORT="${DASHBOARD_PORT:-3000}"
export NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://localhost:8000}"

eval $(minikube docker-env --shell='bash')

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   AutoML Platform Launcher   ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed. Please install it first.${NC}"
    exit 1
fi

# Check for npm (Next.js dashboard)
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed. Please install Node.js/npm to run the dashboard.${NC}"
    exit 1
fi

PID_SAVE_LOCATION='/tmp/pids.txt'
clear_pids() {
    echo '' > "$PID_SAVE_LOCATION"
}

save_pid() {
    echo "$1" >> "$PID_SAVE_LOCATION"
}

recurse_kill() {
    local pid=$1

    if [[ -z "$pid" ]] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        return
    fi

    local child_pids=$(pgrep -P "$pid")
    for child in $child_pids; do
        recurse_kill "$child"
    done

    kill -9 "$pid" 2>/dev/null
}

kill_pids() {
    [ -e "$PID_SAVE_LOCATION" ] || return
    while IFS= read -r pid; do
        [ -n "$pid" ] && recurse_kill "$pid"
    done < "$PID_SAVE_LOCATION"
}

force_kill_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        local pids=$(lsof -t -i:$port 2>/dev/null)
        if [ -n "$pids" ]; then
            echo -e "${BLUE}Force killing processes on port $port...${NC}"
            echo "$pids" | xargs kill -9 2>/dev/null
        fi
    fi
}

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${BLUE}Cleanup...${NC}"
    kill_pids

    force_kill_port 5001
    force_kill_port 8000
    force_kill_port "$FRONTEND_PORT"
}

cleanup

[ "$1" = '--stop' ] && exit

clear_pids

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
uv run mlflow server --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5001 --serve-artifacts > /dev/null 2>&1 &
MLFLOW_PID=$!
save_pid "$MLFLOW_PID"

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
uv run uvicorn src.api.main:app --reload --port 8000 > /dev/null 2>&1 &
API_PID=$!
save_pid "$API_PID"

sleep 2

echo -e "${GREEN}[3/4] Starting Job Worker...${NC}"
if [ -z "$IP_ADDR"]; then
    IP_ADDR=$(
        if [[ $(uname) == "Darwin" ]]; then
            ipconfig getifaddr en0
        else
            ip route get 1.1.1.1 | awk '{print $7; exit}'
        fi
    )
fi
export IP_ADDR
uv run python main.py &
WORKER_PID=$!
save_pid "$WORKER_PID"

echo -e "${GREEN}[4/4] Starting Next.js Dashboard...${NC}"
(
    cd "$FRONTEND_DIR" || exit 1
    if [ ! -d node_modules ]; then
        echo -e "${BLUE}Installing dashboard dependencies...${NC}"
        npm install
    fi
    npm run dev -- --hostname 0.0.0.0 --port "$FRONTEND_PORT"
) > /dev/null 2>&1 &
DASHBOARD_PID=$!
save_pid "$DASHBOARD_PID"

sleep 2

clear

echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}Platform is running!${NC}"
echo -e "MLflow:    http://localhost:5001"
echo -e "API:       http://localhost:8000"
echo -e "Dashboard: http://localhost:${FRONTEND_PORT}"
echo -e "${BLUE}=========================================${NC}"
echo -e "Press Ctrl+C to stop all services."
