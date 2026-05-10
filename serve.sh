#!/bin/bash

PORT=1234

# Hàm dọn dẹp khi thoát (Ctrl+C)
cleanup() {
    echo -e "\n🛑 Stopping server on port $PORT..."
    fuser -k $PORT/tcp 2>/dev/null
    exit
}

# Đăng ký trap
trap cleanup SIGINT SIGTERM

echo "🚀 Starting BetaZero Visualizer at http://localhost:$PORT/and_or_graph.html"
python3 -m http.server --directory . $PORT &
SERVER_PID=$!

# Đợi một chút cho server khởi động rồi mới mở browser
sleep 1
if command -v xdg-open > /dev/null; then
    xdg-open "http://localhost:$PORT/and_or_graph.html"
else
    echo "🔗 Open this link in your browser: http://localhost:$PORT/and_or_graph.html"
fi

# Giữ script chạy để trap hoạt động
wait $SERVER_PID