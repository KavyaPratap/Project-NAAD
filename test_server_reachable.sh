#!/usr/bin/env bash
# Quick connectivity self-test for Project NAAD server
# Run: bash test_server_reachable.sh
# Keep ./run_server.sh running in another terminal first!

echo "=== Project NAAD — Port 5555 Reachability Test ==="
echo ""

LAPTOP_IP="192.168.1.128"
PORT=5555

echo "1. Checking if port $PORT is actually bound..."
if ss -tlnp | grep -q ":$PORT"; then
    PROC=$(ss -tlnp | grep ":$PORT" | grep -o '"[^"]*"' | head -1)
    echo "   ✅ Port $PORT is LISTENING (process: $PROC)"
else
    echo "   ❌ Port $PORT is NOT listening — start ./run_server.sh first!"
    exit 1
fi

echo ""
echo "2. Testing loopback (127.0.0.1) connection..."
if nc -z -w2 127.0.0.1 $PORT 2>/dev/null; then
    echo "   ✅ Loopback OK"
else
    echo "   ❌ Loopback FAILED — something is very wrong with the server"
fi

echo ""
echo "3. Testing WiFi interface ($LAPTOP_IP) connection..."
if nc -z -w2 $LAPTOP_IP $PORT 2>/dev/null; then
    echo "   ✅ WiFi interface OK — ESP32 CAN reach the server!"
    echo ""
    echo "   🎉 Everything is good. Flash and test the ESP32."
else
    echo "   ❌ WiFi interface BLOCKED — ESP32 cannot reach server!"
    echo ""
    echo "   Fix options (run ONE of these):"
    echo ""
    echo "   Option A — Use firewalld (recommended):"
    echo "     sudo systemctl start firewalld"
    echo "     sudo firewall-cmd --zone=public --add-port=$PORT/tcp"
    echo "     sudo firewall-cmd --zone=public --add-port=$PORT/tcp --permanent"
    echo ""
    echo "   Option B — Direct nftables rule:"
    echo "     sudo nft add rule inet firewalld filter_IN_public_allow tcp dport $PORT accept"
    echo ""
    echo "   Option C — Flush all firewall rules (dev machine only):"
    echo "     sudo nft flush ruleset"
    echo ""
    echo "   After applying fix, re-run this script to verify."
fi
