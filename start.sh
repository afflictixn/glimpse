#!/bin/bash
# Z Exp — start all services
# Usage: ./start.sh [--debug|-d] [--verbose|-v] [--kill]
#   --debug/-d   : debug logs in terminal/log file only
#   --verbose/-v : debug logs in terminal + overlay sidebar

set -e
cd "$(dirname "$0")"

# Workaround: M5 Metal 4 tensor ops have a bfloat/half type mismatch in
# MetalPerformancePrimitives that crashes ggml shader compilation.
# Disabling tensor ops keeps all other Metal GPU acceleration intact.
export GGML_METAL_TENSOR_DISABLE=1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Kill helper
kill_all() {
    echo -e "${YELLOW}Stopping all Z Exp services...${NC}"
    pkill -f "ollama serve" 2>/dev/null && echo "  ollama stopped" || true
    pkill -f ZExpOverlay 2>/dev/null && echo "  overlay stopped" || true
    pkill -f "src.main" 2>/dev/null && echo "  backend stopped" || true
    echo -e "${GREEN}All stopped.${NC}"
}

# Parse flags (can combine: ./start.sh --debug --gemini)
DEBUG_FLAGS=""
PROVIDER=""
while [[ "$1" == --* ]]; do
    case "$1" in
        --debug|-d)
            DEBUG_FLAGS="--debug"
            echo -e "${YELLOW}Debug mode: debug logs → terminal / log file${NC}"
            shift ;;
        --verbose|-v)
            DEBUG_FLAGS="--debug --debug-ws"
            echo -e "${YELLOW}Verbose mode: debug logs → terminal + overlay sidebar${NC}"
            shift ;;
        --gemini)
            PROVIDER="--provider gemini"
            echo -e "${YELLOW}Provider: Gemini (cloud)${NC}"
            shift ;;
        --ollama|--local)
            PROVIDER="--provider ollama"
            echo -e "${YELLOW}Provider: Ollama (local)${NC}"
            shift ;;
        *) break ;;
    esac
done

# Kill mode
if [ "$1" = "--kill" ] || [ "$1" = "-k" ]; then
    kill_all
    exit 0
fi

# Restart mode (optionally a single service: overlay, backend, ollama)
if [ "$1" = "--restart" ] || [ "$1" = "-r" ]; then
    SERVICE="$2"
    if [ -z "$SERVICE" ]; then
        kill_all
        sleep 2
        echo ""
    else
        case "$SERVICE" in
            overlay|swift)
                echo -e "${YELLOW}Restarting overlay...${NC}"
                pkill -f ZExpOverlay 2>/dev/null || true
                sleep 1
                cd overlay
                swift run ZExpOverlay > /tmp/zexp-overlay.log 2>&1 &
                cd ..
                sleep 3
                if pgrep -f ZExpOverlay > /dev/null 2>&1; then
                    echo -e "  ${GREEN}✓${NC} Overlay restarted"
                else
                    echo -e "  ${RED}✗${NC} Overlay failed (check /tmp/zexp-overlay.log)"
                fi
                exit 0
                ;;
            backend|python)
                echo -e "${YELLOW}Restarting backend...${NC}"
                pkill -f "src.main" 2>/dev/null || true
                sleep 1
                source .venv/bin/activate 2>/dev/null || true
                python -m src.main --port 3030 $DEBUG_FLAGS $PROVIDER > /tmp/zexp-backend.log 2>&1 &
                sleep 2
                if curl -s http://localhost:3030/health > /dev/null 2>&1; then
                    echo -e "  ${GREEN}✓${NC} Backend restarted (port 3030)"
                else
                    echo -e "  ${YELLOW}~${NC} Backend starting... (check /tmp/zexp-backend.log)"
                fi
                exit 0
                ;;
            ollama)
                echo -e "${YELLOW}Restarting Ollama...${NC}"
                pkill -f "ollama serve" 2>/dev/null || true
                sleep 1
                ollama serve > /dev/null 2>&1 &
                sleep 3
                if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
                    echo -e "  ${GREEN}✓${NC} Ollama restarted"
                else
                    echo -e "  ${RED}✗${NC} Ollama failed to start"
                fi
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown service: $SERVICE${NC}"
                echo "  Valid: overlay (swift), backend (python), ollama"
                exit 1
                ;;
        esac
    fi
fi

echo -e "${GREEN}Starting Z Exp...${NC}"

# 1. Ollama (only needed for ollama vision provider)
source .venv/bin/activate 2>/dev/null || true
VISION_PROVIDER=$(python -c "from src.config import Settings; print(Settings().vision_provider)" 2>/dev/null || echo "unknown")
if [ "$VISION_PROVIDER" = "ollama" ]; then
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
else
    echo -e "  ${YELLOW}–${NC} Ollama skipped (vision_provider=${VISION_PROVIDER})"
fi

# 2. Swift overlay (bundled .app for macOS permissions)
if pgrep -f ZExpOverlay > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Overlay already running"
else
    echo -e "  ${YELLOW}→${NC} Building & starting overlay..."
    cd overlay
    bash bundle.sh > /tmp/zexp-overlay-build.log 2>&1
    open .build/ZExpOverlay.app > /tmp/zexp-overlay.log 2>&1
    cd ..
    sleep 3
    if pgrep -f ZExpOverlay > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Overlay started (Cmd+Shift+O to toggle)"
    else
        echo -e "  ${RED}✗${NC} Overlay failed (check /tmp/zexp-overlay.log)"
    fi
fi

# 3. Python backend
if curl -s http://localhost:3030/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Z Exp backend already running"
else
    echo -e "  ${YELLOW}→${NC} Starting Z Exp backend..."
    source .venv/bin/activate 2>/dev/null || true
    python -m src.main --port 3030 $DEBUG_FLAGS $PROVIDER > /tmp/zexp-backend.log 2>&1 &
    sleep 2
    if curl -s http://localhost:3030/health > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Z Exp backend started (port 3030)"
    else
        echo -e "  ${YELLOW}~${NC} Z Exp backend starting... (check /tmp/zexp-backend.log)"
    fi
fi

echo ""
echo -e "${GREEN}Z Exp is running.${NC}"
echo "  Ollama:   http://localhost:11434"
echo "  Backend:  http://localhost:3030"
echo "  Overlay:  Cmd+Shift+O to toggle"
echo "  WebSocket: ws://localhost:3030/ws"
echo ""
echo "  Stop all: ./start.sh --kill"
