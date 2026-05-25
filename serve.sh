#!/bin/bash

PORT=1234

cleanup() {
    echo -e "\nStopping server on port $PORT..."
    fuser -k $PORT/tcp 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

echo "Starting BetaZero Visualizer at http://localhost:$PORT/and_or_graph.html"
python3 -m http.server --directory . $PORT &
SERVER_PID=$!

sleep 1
if command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:$PORT/and_or_graph.html"
else
    echo "🔗 Open this link in your browser: http://localhost:$PORT/and_or_graph.html"
fi

wait $SERVER_PID