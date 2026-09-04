#!/usr/bin/env bash
# Project NAAD — Flash + Monitor Helper
# Usage: ./flash_and_monitor.sh [PORT]
# Default port: /dev/ttyACM0
#
# FIRST-TIME SETUP (run once, then logout + login):
#   sudo usermod -aG dialout $USER
#
# Prerequisites:
#   source ~/esp/esp-idf/export.sh    (or add to your ~/.bashrc)

set -e
PORT="${1:-/dev/ttyACM0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIRMWARE_DIR="$SCRIPT_DIR/firmware"

echo "==================================================="
echo "  Project NAAD — Flash & Monitor"
echo "  Port     : $PORT"
echo "  Firmware : $FIRMWARE_DIR"
echo "==================================================="
echo ""

# Check port is accessible
if [ ! -r "$PORT" ]; then
    echo "❌ Cannot read $PORT"
    echo "   Fix: sudo usermod -aG dialout $USER   then logout + login"
    echo "   Or:  sudo chmod a+rw $PORT  (temporary, for current session)"
    exit 1
fi

cd "$FIRMWARE_DIR"
echo "⚙️  Building..."
idf.py build

echo ""
echo "📦 Flashing to $PORT ..."
idf.py -p "$PORT" flash

echo ""
echo "📡 Starting monitor (Ctrl+] to exit)..."
idf.py -p "$PORT" monitor
