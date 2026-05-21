#!/usr/bin/env bash

N=${1:-$(date +%Y%m%d)}
threads=${2:-32}
cache_budget=${3:-}
threads_safe=${threads//,/_}

# Check if N is NULL (no bandwidth limit)
if [[ "$N" == "NULL" || "$N" == "null" ]]; then
    echo "N is NULL, skipping bandwidth limit"
    SKIP_FIO_LIMIT=true
elif ! [[ "$N" =~ ^([1-9][0-9]?|100)$ ]]; then
    echo "Warning: N='$N' is not a valid 1-100 integer or NULL."
    while true; do
        read -p "Enter a value 1-100 or NULL: " N
        if [[ "$N" == "NULL" || "$N" == "null" ]]; then
            echo "Set N = NULL, skipping bandwidth limit"
            SKIP_FIO_LIMIT=true
            break
        elif [[ "$N" =~ ^([1-9][0-9]?|100)$ ]]; then
            echo "Set N = $N"
            SKIP_FIO_LIMIT=false
            break
        else
            echo "Error: '$N' is not in range 1-100 or NULL."
        fi
    done
else
    SKIP_FIO_LIMIT=false
fi

echo "N: $N"
echo "threads: $threads"
echo "cache_budget: ${cache_budget:-disabled}"
echo "threads_safe: $threads_safe"

START_TIME=$(date +%s)
START_FMT=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start: $START_FMT"
echo "========================================"

LOG_DIR="/mnt/data/bench_logs/${N}_threads_${threads_safe}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Log dir: $LOG_DIR"

if [[ "$SKIP_FIO_LIMIT" == "true" ]]; then
    source mv_shell_back.sh
else
    source set_fio_limit_v2.sh /dev/nvme2n1 ${N}
    source mv_shell_to_fio_limit_v2.sh
fi

cat /proc/$$/cgroup | grep blkio
sleep 16

bash ssd-conc.sh /dev/nvme2n1 5 8k

bash test_100m.sh search ${threads} ${cache_budget} 2>&1 | tee -a "$LOG_DIR/Fast_100m.txt"
bash test_bge.sh search ${threads} ${cache_budget} 2>&1 | tee -a "$LOG_DIR/Fast_bge.txt"

END_TIME=$(date +%s)
END_FMT=$(date "+%Y-%m-%d %H:%M:%S")
DURATION=$((END_TIME - START_TIME))
H=$((DURATION / 3600))
M=$(((DURATION % 3600) / 60))
S=$((DURATION % 60))

echo "========================================"
echo "End: $END_FMT"
echo "Duration: ${H}h${M}m${S}s (${DURATION}s total)"
echo "Logs: $LOG_DIR"

echo "N=$N, threads=$threads, cache_budget=${cache_budget:-none}, start=$START_FMT, end=$END_FMT, duration=${DURATION}s" >> "$LOG_DIR/summary.txt"
