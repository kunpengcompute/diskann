#!/usr/bin/env bash

N=${1:-$(date +%Y%m%d)}
threads=${2:-32}
threads_safe=${threads//,/_}

# 检查是否为特殊值NULL（不限制带宽）
if [[ "$N" == "NULL" || "$N" == "null" ]]; then
    echo "✅ N 值为 NULL，跳过带宽限制设置"
    SKIP_FIO_LIMIT=true
elif ! [[ "$N" =~ ^([1-9][0-9]?|100)$ ]]; then
    echo "⚠️  当前的 N 值 '$N' 不是有效的 1-100 整数或 NULL。"
    while true; do
        read -p "请设定一个 1-100 的整数值或 NULL: " N
        if [[ "$N" == "NULL" || "$N" == "null" ]]; then
            echo "✅ 已设置 N = NULL，跳过带宽限制设置"
            SKIP_FIO_LIMIT=true
            break
        elif [[ "$N" =~ ^([1-9][0-9]?|100)$ ]]; then
            echo "✅ 已设置 N = $N"
            SKIP_FIO_LIMIT=false
            break
        else
            echo "❌  '$N' 不在 1-100 范围内且不是 NULL，请重新输入。"
        fi
    done
else
    SKIP_FIO_LIMIT=false
fi

echo "当前 N 值为: $N"
echo "当前 threads 值为: $threads"
echo "当前 threads_safe 值为: $threads_safe"

# 记录开始时间
START_TIME=$(date +%s)
START_FMT=$(date "+%Y-%m-%d %H:%M:%S")
echo "脚本开始运行: $START_FMT"
echo "========================================"

# 创建统一的日志目录，避免文件分散
LOG_DIR="/mnt/data/bench_logs/${N}_threads_${threads_safe}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "日志目录: $LOG_DIR"

# 根据N值决定是否设置带宽限制
if [[ "$SKIP_FIO_LIMIT" == "true" ]]; then
    echo "跳过带宽限制设置，执行 mv_shell_back.sh"
    source mv_shell_back.sh
else
    source set_fio_limit_v2.sh /dev/nvme2n1 ${N}
    echo "设置带宽限制，执行 mv_shell_to_fio_limit_v2.sh"
    source mv_shell_to_fio_limit_v2.sh
fi

cat /proc/$$/cgroup | grep blkio
sleep 16

bash ssd-conc.sh /dev/nvme2n1 5 8k

bash test_100m.sh search ${threads} 2>&1 | tee -a "$LOG_DIR/Fast_100m.txt"
bash test_bge.sh search ${threads} 2>&1 | tee -a "$LOG_DIR/Fast_bge.txt"

# 记录结束时间并计算时长
END_TIME=$(date +%s)
END_FMT=$(date "+%Y-%m-%d %H:%M:%S")
DURATION=$((END_TIME - START_TIME))
H=$((DURATION / 3600))
M=$(((DURATION % 3600) / 60))
S=$((DURATION % 60))

echo "========================================"
echo "🏁 脚本结束运行: $END_FMT"
echo "⏱️  总运行时长: ${H}小时${M}分${S}秒 (共${DURATION}秒)"
echo "📂 日志保存在: $LOG_DIR"

# 保存运行摘要
echo "N=$N, threads=$threads, 开始=$START_FMT, 结束=$END_FMT, 时长=${DURATION}秒" >> "$LOG_DIR/summary.txt"