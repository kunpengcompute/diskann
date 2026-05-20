#!/bin/bash
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_dir> [build|search|search_cache|all]"
    echo "  data_dir: directory containing sift_learn.fbin and sift_query.fbin"
    echo "Example: $0 /mnt/data/sift all"
    exit 1
fi

DATA_DIR=$1
MODE=$2
CURRENT_DIR=$(pwd)
INDEX_DIR=${DATA_DIR}/index
BUILD_APP=$CURRENT_DIR/../build/apps/build_disk_index
SEARCH_APP=$CURRENT_DIR/../build/apps/search_disk_index
GT_APP=$CURRENT_DIR/../build/apps/utils/compute_groundtruth

BASE_FILE=${DATA_DIR}/sift_learn.fbin
QUERY_FILE=${DATA_DIR}/sift_query.fbin
GT_FILE=${DATA_DIR}/sift_query_learn_gt100
INDEX_PREFIX=${INDEX_DIR}/sift_R64_L100

mkdir -p $INDEX_DIR

compute_gt() {
    if [ -f "$GT_FILE" ]; then
        echo "Ground truth file exists, skipping."
        return
    fi
    echo "Computing ground truth..."
    $GT_APP --data_type float --dist_fn l2 \
        --base_file $BASE_FILE \
        --query_file $QUERY_FILE \
        --gt_file $GT_FILE --K 100
}

build() {
    echo "Building disk index..."
    $BUILD_APP --data_type float --dist_fn l2 \
        --data_path $BASE_FILE \
        --index_path_prefix $INDEX_PREFIX \
        -R 64 -L 100 -M 1 -T 16 --QD 64 \
        --search_DRAM_budget 0.5 --generate_mem_file true
}

search() {
    echo "Searching..."
    mkdir -p ${INDEX_DIR}/result/
    $SEARCH_APP --data_type float --dist_fn l2 \
        --index_path_prefix $INDEX_PREFIX \
        --query_file $QUERY_FILE \
        --gt_file $GT_FILE \
        -K 10 -L 10 20 30 40 50 100 \
        --result_path ${INDEX_DIR}/result/ \
        -T 16 -W 4 \
        --memory_graph_path ${INDEX_PREFIX}_mem.index.vamana.comp \
        --reorder_ratio 0.90
}

search_cache() {
    echo "Searching with cache..."
    mkdir -p ${INDEX_DIR}/result_cache/
    $SEARCH_APP --data_type float --dist_fn l2 \
        --index_path_prefix $INDEX_PREFIX \
        --query_file $QUERY_FILE \
        --gt_file $GT_FILE \
        -K 10 -L 10 20 30 40 50 100 \
        --result_path ${INDEX_DIR}/result_cache/ \
        -T 16 -W 4 \
        --reorder_ratio 0.90 \
        --cache_budget 0.5 \
        --repeat 3
}

case "$MODE" in
    "build")
        compute_gt
        build
        ;;
    "search")
        search
        ;;
    "search_cache")
        search_cache
        ;;
    "all")
        compute_gt
        build
        search
        search_cache
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac
