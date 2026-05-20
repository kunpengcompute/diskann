#!/bin/bash
if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments."
    echo "Usage: $0 buildIndex [threads]"
    echo "       $0 search [threads] [cache_budget_gb]"
    echo "Example: $0 search 16,32"
    echo "         $0 search 48 2"
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
        --search_DRAM_budget 5 --generate_mem_file true
}

search() {
    CACHE_BUDGET=$1

    data_type="float"
    dist_fn="l2"
    index_path_prefix="$INDEX_DIR/10M_1024_R64_L100QD128"
    query_file="$DATA_DIR/query.bin"
    gt_file="$DATA_DIR/gt.bin"
    K=10

    IFS=',' read -ra threads <<< "$THREADS_INPUT"
    Reranks=(0.80 0.85 0.90 0.95)

    if [ -n "$CACHE_BUDGET" ]; then
        result_path="./result/10M1024D_cache/"
        beams=(2 4 8 16)
    else
        result_path="./result/10M1024D/"
        beams=(2 3 4 6 8 12 16 24 32)
    fi
    mkdir -p $result_path

    for T in "${threads[@]}"; do
        for Rerank in "${Reranks[@]}"; do
            for W in "${beams[@]}"; do
                if [ "$Rerank" == "0.95" ]; then
                    L="500 600 700 800 900 1000"
                else
                    L="200 250 300 350 400 450 500"
                fi

                cmd="numactl --physcpubind=0-$((T-1)) --membind=0 \
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
                    --reorder_ratio $Rerank \
                    -W $W"

                if [ -n "$CACHE_BUDGET" ]; then
                    cmd="$cmd \
                    --cache_budget $CACHE_BUDGET \
                    --repeat 3"
                else
                    cmd="$cmd \
                    --memory_graph_path $INDEX_DIR/10M_1024_R64_L100QD128_mem.index.vamana.comp"
                fi

                echo "Executing command: $cmd"
                eval "$cmd"
            done
        done
    done
}

case "$1" in
    "buildIndex")
        buildIndex
        ;;
    "search")
        echo "Running search test..."
        search "$3"
        ;;
    *)
        exit 1
        ;;
esac
