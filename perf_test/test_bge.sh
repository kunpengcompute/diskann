#!/bin/bash
if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments."
    echo "Usage: $0 buildIndex [threads]"
    echo "       $0 search [threads]"
    echo "Example: $0 search 16,32"
    exit 1
fi

MODE=$1
THREADS_INPUT=$2
CURRENT_DIR=$(pwd)
INDEX_DIR=/mnt/data/diakann/index/Fast/bge_10m_1024
DATA_DIR=/mnt/data/diskann/data/bge_10m_1024

buildIndex() {
    echo "Building Index..."
    $CURRENT_DIR/../build/apps/build_disk_index \
        --dist_fn l2 \
        --data_path ${DATA_DIR}/base.bin \
        --index_path_prefix ${INDEX_DIR}/10M_1024_R64_L100QD128 \
        -R 64 -L 100 -M 5 -T 160 --QD 128 --data_type float \
        --search_DRAM_budget 5
}

search() {
    data_type="float"
    dist_fn="l2"
    index_path_prefix="$INDEX_DIR/10M_1024_R64_L100QD128"
    query_file="$DATA_DIR/query.bin"
    gt_file="$DATA_DIR/gt.bin"
    K=10

    IFS=',' read -ra threads <<< "$THREADS_INPUT"
    result_path="./result/10M1024D/"
    beams=(2 3 4 6 8 12 16 24 32)
    mkdir -p $result_path

    for T in "${threads[@]}"; do
        for W in "${beams[@]}"; do
            L="200 250 300 350 400 450 500"

            cmd="numactl -N 0 -m 0 \
                $CURRENT_DIR/../build/apps/search_disk_index \
                --data_type $data_type \
                --dist_fn $dist_fn \
                --index_path_prefix $index_path_prefix \
                --query_file $query_file \
                --gt_file $gt_file \
                -K $K \
                -L $L \
                --result_path $result_path \
                -T $T \
                -W $W"

            echo "Executing command: $cmd"
            eval "$cmd"
        done
    done
}

case "$1" in
    "buildIndex")
        buildIndex
        ;;
    "search")
        echo "Running search test..."
        search
        ;;
    *)
        exit 1
        ;;
esac
