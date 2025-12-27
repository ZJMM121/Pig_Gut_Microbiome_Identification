#!/usr/bin/env bash

# 训练监控脚本

SCRIPT_DIR=$(dirname $(readlink -f $0))
LOG_DIR=$SCRIPT_DIR"/logs"

if [ $# -eq 0 ]; then
    echo "用法: $0 <时间戳>"
    echo "可用的训练会话:"
    ls -la $LOG_DIR/training_*.status 2>/dev/null | head -10
    exit 1
fi

TIMESTAMP=$1
LOG_FILE=$LOG_DIR"/training_${TIMESTAMP}.log"
PID_FILE=$LOG_DIR"/training_${TIMESTAMP}.pid"
STATUS_FILE=$LOG_DIR"/training_${TIMESTAMP}.status"

echo "==================== 训练状态检查 ===================="
echo "时间戳: $TIMESTAMP"
echo "日志文件: $LOG_FILE"

# 检查状态文件
if [ -f "$STATUS_FILE" ]; then
    echo ""
    echo "📊 状态信息:"
    cat $STATUS_FILE
    echo ""
else
    echo "❌ 状态文件不存在: $STATUS_FILE"
    exit 1
fi

# 检查PID文件和进程状态
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    echo "🔍 进程检查:"
    echo "PID: $PID"
    
    if kill -0 $PID 2>/dev/null; then
        echo "状态: ✅ 运行中"
        
        # 显示资源使用情况
        echo ""
        echo "📈 资源使用:"
        ps -p $PID -o pid,ppid,cmd,%cpu,%mem,etime --no-headers 2>/dev/null || echo "无法获取进程信息"
        
    else
        echo "状态: ❌ 已停止"
    fi
    echo ""
else
    echo "❌ PID文件不存在: $PID_FILE"
fi

# 检查日志文件
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo "0")
    LOG_SIZE_MB=$((LOG_SIZE / 1024 / 1024))
    LOG_LINES=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
    
    echo "📄 日志信息:"
    echo "文件大小: ${LOG_SIZE_MB} MB"
    echo "行数: $LOG_LINES"
    echo ""
    
    echo "📝 最新日志 (最后10行):"
    tail -10 "$LOG_FILE" 2>/dev/null || echo "无法读取日志文件"
    echo ""
    
    # 检查是否有loss输出
    LOSS_COUNT=$(grep -c "loss:" "$LOG_FILE" 2>/dev/null || echo "0")
    if [ $LOSS_COUNT -gt 0 ]; then
        echo "🎯 训练进度:"
        echo "Loss报告次数: $LOSS_COUNT"
        echo "最新Loss值:"
        grep "loss:" "$LOG_FILE" | tail -3 2>/dev/null || echo "无loss信息"
    else
        echo "⏳ 还未开始loss报告"
    fi
    
else
    echo "❌ 日志文件不存在: $LOG_FILE"
fi

echo ""
echo "🔄 实时监控命令:"
echo "  tail -f $LOG_FILE"
echo ""
echo "🛑 停止训练命令:"
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    echo "  kill $PID"
fi
