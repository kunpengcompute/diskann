#!/bin/bash
if [ $# -lt 2 ]; then
    echo "Error: Insufficient arguments."
    echo "Usage: $0 buildIndex [threads]"
    echo "       $0 search [threads]"
    echo "       $0 search_cache [threads] [cache_budget_gb]"
    echo "Example: $0 search 16,32"
    exit 1
fi

MODE=$1
THREADS_INPUT=$2
CURRENT_DIR=$(pwd)
INDEX_DIR=/mnt/data/diskann/index/100m_1536
DATA_DIR=/mnt/data/diskann/data/100M_1536

buildIndex() {
    echo "Building Index..."
    $CURRENT_DIR/../build/apps/build_disk_index \
        --dist_fn l2 \
        --data_path ${DATA_DIR}/base_fp32.bin \
        --index_path_prefix ${INDEX_DIR}/100M_1536_R64_L100QD192 \
        -R 64 -L 100 -M 80 -T 80 --QD 192 --data_type float \
        --search_DRAM_budget 1 --generate_mem_file true
}

search() {
    data_type="float"
    dist_fn="l2"
    index_path_prefix="$INDEX_DIR/100M_1536_R64_L100QD192"
    query_file="$DATA_DIR/query.bin"
    gt_file="$DATA_DIR/gt.bin"
    result_path="./result/100M1536D/"
    mkdir -p $result_path

    K=10
    L="300 400 500 600 800 1000"

    IFS=',' read -ra threads <<< "$THREADS_INPUT"
    beams=(2 3 4 6 8 12 16 24 32)
    Reranks=(0.80 0.85 0.90 0.95)

    for T in "${threads[@]}"; do
        for Rerank in "${Reranks[@]}"; do
            for W in "${beams[@]}"; do
                if [ "$Rerank" == "0.95" ]; then
                    L="500 600 700 800 900 1000"
                else
                    L="150 200 250 300 350 400 450 500"
                fi

                cmd="numactl -N 0 --membind=0 \
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
                    --memory_graph_path $INDEX_DIR/100M_1536_R64_L100QD192_mem.index.vamana.comp \
                    --reorder_ratio $Rerank \
                    -W $W"

                echo "Executing command: $cmd"
                eval "$cmd"
            done
        done
    done
}

search_cache() {
    THREADS_INPUT=$1
    CACHE_BUDGET=${2:-4}

    data_type="float"
    dist_fn="l2"
    index_path_prefix="$INDEX_DIR/100M_1536_R64_L100QD192"
    query_file="$DATA_DIR/query.bin"
    gt_file="$DATA_DIR/gt.bin"
    result_path="./result/100M1536D_cache/"
    priority_file="$INDEX_DIR/100M_1536_R64_L100QD192_mem.index.priority.indegree"
    mkdir -p $result_path

    K=10
    IFS=',' read -ra threads <<< "$THREADS_INPUT"
    beams=(2 4 8 16 32)
    Reranks=(0.80 0.85 0.90 0.95)

    for T in "${threads[@]}"; do
        for Rerank in "${Reranks[@]}"; do
            for W in "${beams[@]}"; do
                if [ "$Rerank" == "0.95" ]; then
                    L="500 600 700 800 900 1000"
                else
                    L="150 200 250 300 350 400 450 500"
                fi

                cmd="numactl -N 0 --membind=0 \
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
                    --memory_graph_path $INDEX_DIR/100M_1536_R64_L100QD192_mem.index.vamana.comp \
                    --reorder_ratio $Rerank \
                    -W $W \
                    --cache_budget $CACHE_BUDGET \
                    --graph_priority_file $priority_file \
                    --repeat 3"

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
        echo "Running searching test..."
        search
        ;;
    "search_cache")
        echo "Running cache search test (budget=${3:-4}GB)..."
        search_cache "$2" "$3"
        ;;
    *)
        exit 1
        ;;
esac