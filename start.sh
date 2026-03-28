#!/bin/bash
# Glimpse — start all services
# Usage: ./start.sh [--kill]

set -e
cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Kill mode
if [ "$1" = "--kill" ] || [ "$1" = "-k" ]; then
    echo -e "${YELLOW}Stopping all Glimpse services...${NC}"
    pkill -f "ollama serve" 2>/dev/null && echo "  ollama stopped" || true
    pkill -f GlimpseOverlay 2>/dev/null && echo "  overlay stopped" || true
    pkill -f "src.main" 2>/dev/null && echo "  backend stopped" || true
    echo -e "${GREEN}All stopped.${NC}"
    exit 0
fi

echo -e "${GREEN}Starting Glimpse...${NC}"

# 1. Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Ollama already running"
else
    echo -e "  ${YELLOW}→${NC} Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Ollama started"
    else
        echo -e "  ${RED}✗${NC} Ollama failed to start"
    fi
fi

# 2. Swift overlay (bundled .app for macOS permissions)
if pgrep -f GlimpseOverlay > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Overlay already running"
else
    echo -e "  ${YELLOW}→${NC} Building & starting overlay..."
    cd overlay
    bash bundle.sh > /tmp/glimpse-overlay-build.log 2>&1
    open .build/GlimpseOverlay.app > /tmp/glimpse-overlay.log 2>&1
    cd ..
    sleep 3
    if pgrep -f GlimpseOverlay > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Overlay started (Cmd+Shift+O to toggle)"
    else
        echo -e "  ${RED}✗${NC} Overlay failed (check /tmp/glimpse-overlay.log)"
    fi
fi

# 3. Python backend
if curl -s http://localhost:3030/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Glimpse backend already running"
else
    echo -e "  ${YELLOW}→${NC} Starting Glimpse backend..."
    source .venv/bin/activate 2>/dev/null || true
    python -m src.main --port 3030 > /tmp/glimpse-backend.log 2>&1 &
    sleep 2
    if curl -s http://localhost:3030/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Glimpse backend started (port 3030)"
    else
        echo -e "  ${YELLOW}~${NC} Glimpse backend starting... (check /tmp/glimpse-backend.log)"
    fi
fi

echo ""
echo -e "${GREEN}Glimpse is running.${NC}"
echo "  Ollama:   http://localhost:11434"
echo "  Backend:  http://localhost:3030"
echo "  Overlay:  Cmd+Shift+O to toggle"
echo "  WebSocket: ws://localhost:9321"
echo ""
echo "  Stop all: ./start.sh --kill"
