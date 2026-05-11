#!/bin/bash
# Generate real test index using DiskANN tools

set -e

PREFIX="/tmp/test_real_index"
NUM_POINTS=50
DIM=8
NUM_CHUNKS=2

echo "=== Generating test index for unit tests ==="

# Step 1: Generate random test data
echo "Step 1: Generating test data (${NUM_POINTS} points, ${DIM} dims)..."
python3 << EOF
import struct
import numpy as np

num_points = ${NUM_POINTS}
dim = ${DIM}

# Generate random data
data = np.random.randn(num_points, dim).astype(np.float32)

# Save in DiskANN bin format
with open("${PREFIX}_data.bin", "wb") as f:
    f.write(struct.pack('I', num_points))
    f.write(struct.pack('I', dim))
    data.tofile(f)

print(f"Generated ${PREFIX}_data.bin")
EOF

# Step 2: Generate PQ pivots using DiskANN tool
echo "Step 2: Generating PQ pivots..."
./build/apps/utils/generate_pq \
    --data_type float \
    --data_path ${PREFIX}_data.bin \
    --num_pq_centers 256 \
    --num_pq_chunks ${NUM_CHUNKS} \
    --output_prefix ${PREFIX}

# Step 3: Generate PQ compressed data
echo "Step 3: Generating PQ compressed data..."
python3 << EOF
import struct
import numpy as np

num_points = ${NUM_POINTS}
num_chunks = ${NUM_CHUNKS}

# Generate dummy compressed data (just random chunk IDs)
compressed = np.random.randint(0, 256, size=(num_points, num_chunks), dtype=np.uint8)

with open("${PREFIX}_pq_compressed.bin", "wb") as f:
    f.write(struct.pack('I', num_points))
    f.write(struct.pack('I', num_chunks))
    compressed.tofile(f)

print(f"Generated ${PREFIX}_pq_compressed.bin")
EOF

# Step 4: Create disk index metadata
echo "Step 4: Creating disk index metadata..."
python3 << EOF
import struct

with open("${PREFIX}_disk.index", "wb") as f:
    # Header
    f.write(struct.pack('I', 5))  # nr
    f.write(struct.pack('I', 1))  # nc

    # Metadata
    f.write(struct.pack('Q', ${NUM_POINTS}))  # disk_nnodes
    f.write(struct.pack('Q', ${DIM}))  # disk_ndims
    f.write(struct.pack('Q', 0))  # medoid_id
    f.write(struct.pack('Q', 1024))  # max_node_len
    f.write(struct.pack('Q', 1))  # nnodes_per_sector
    f.write(struct.pack('Q', 0))  # num_frozen
    f.write(struct.pack('Q', 0))  # frozen_id
    f.write(struct.pack('Q', 0))  # reorder_data_exists

print(f"Generated ${PREFIX}_disk.index")
EOF

# Step 5: Create medoids file
echo "Step 5: Creating medoids file..."
python3 << EOF
import struct

with open("${PREFIX}_disk.index_medoids.bin", "wb") as f:
    f.write(struct.pack('I', 1))  # npts
    f.write(struct.pack('I', 1))  # ndims
    f.write(struct.pack('I', 0))  # medoid id = 0

print(f"Generated ${PREFIX}_disk.index_medoids.bin")
EOF

echo "=== Test index generation complete ==="
echo "Files generated:"
ls -lh ${PREFIX}* | awk '{print $9, $5}'
