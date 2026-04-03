#!/bin/bash
# LSTM训练进度监控脚本 - 每10分钟汇报一次

LOG_FILE="/home/zzy/project/ASSTS/ASSTS-stock_class/lstm_stage/train.log"
PID_FILE="/home/zzy/project/ASSTS/ASSTS-stock_class/lstm_stage/train.pid"
CHECKPOINT_DIR="/home/zzy/project/ASSTS/ASSTS-stock_class/lstm_stage/checkpoints"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控启动"
echo "PID: $(cat $PID_FILE 2>/dev/null || echo 'N/A')"

while true; do
    sleep 600  # 10分钟
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat $PID_FILE)
        if ! kill -0 $PID 2>/dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 训练进程已结束"
            echo "=== 最终日志 ==="
            tail -20 "$LOG_FILE"
            
            # 检查是否有checkpoint
            if [ -f "$CHECKPOINT_DIR/best_model.pth" ]; then
                echo "=== 模型已保存 ==="
                ls -lh "$CHECKPOINT_DIR/"
            fi
            break
        fi
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 进度汇报:"
    tail -5 "$LOG_FILE" 2>/dev/null
    echo "---"
done
