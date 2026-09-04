#!/usr/bin/env bash
# Project NAAD — ASR Server Launcher
# Run from project root: ./run_server.sh
# Make executable: chmod +x run_server.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="$SCRIPT_DIR/.venv_server/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "❌ venv not found at .venv_server/"
    echo "   Create it with: python3 -m venv .venv_server"
    echo "   Then install deps: .venv_server/bin/pip install faster-whisper numpy"
    exit 1
fi

echo "==================================================="
echo "  Project NAAD — ASR Server"
echo "  Python : $PYTHON"
echo "  Server : server/receiver.py"
echo "  Port   : 5555"
echo "==================================================="
echo ""

exec "$PYTHON" server/receiver.py
