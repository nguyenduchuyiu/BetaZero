#!/bin/bash

set -o pipefail

# --- CẤU HÌNH ---
THRESHOLD=500
INTERVAL=30
CONDA_ENV="betaproof"
# TRAIN_CMD="python -u train.py configs/deepseek_r1_distill_7B.yaml"
TRAIN_CMD="python -u train_sft.py"
LOG_FILE="gpu_watcher.log"

log_message() {
    local MSG="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$MSG" | tee -a "$LOG_FILE"
}

echo "--- Bắt đầu watcher ---" | tee -a "$LOG_FILE"

while true; do
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    if [ "$GPU_MEM_USED" -lt "$THRESHOLD" ]; then
        log_message "GPU rảnh (${GPU_MEM_USED}MB). Kích hoạt $CONDA_ENV và chạy train..."
        
        # Giải thích:
        # 1. PYTHONUNBUFFERED=1: Ép Python in log ngay lập tức, không đợi đầy buffer
        # 2. conda run --no-capture-output: Đẩy thẳng stdout/stderr ra ngoài shell
        # 3. 2>&1: Gộp cả lỗi (stderr) vào luồng output chung
        # 4. tee -a: Vừa in ra màn hình vừa ghi vào file
        
        export PYTHONUNBUFFERED=1
        conda run --no-capture-output -n "$CONDA_ENV" $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"
        TRAIN_EXIT_CODE=${PIPESTATUS[0]}

        if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
            log_message "--- Quá trình train kết thúc thành công ---"
            break
        fi

        log_message "--- Train lỗi với exit code $TRAIN_EXIT_CODE. Sẽ thử lại sau ${INTERVAL}s ---"
    else
        echo "[$(date '+%H:%M:%S')] GPU bận (${GPU_MEM_USED}MB). Đợi..."
    fi

    sleep "$INTERVAL"
done
