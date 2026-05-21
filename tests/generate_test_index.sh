#!/bin/bash
# Pre-generate test index files to avoid slow k-means during tests

PREFIX="/tmp/test_prebuilt_index"

# Generate minimal PQ pivots (256 centers, 2 chunks, 8 dims)
python3 << 'EOF'
import struct
import numpy as np

prefix = "/tmp/test_prebuilt_index"
num_centers = 256
dim = 8
num_chunks = 2

# Generate random centroids
centroids = np.random.randn(num_centers, dim).astype(np.float32)

# Write PQ pivots file
with open(f"{prefix}_pq_pivots.bin", "wb") as f:
    # Header
    f.write(struct.pack('I', 4))  # nr = 4 sections
    f.write(struct.pack('I', 1))  # nc = 1

    # Section 1: centroids
    f.write(struct.pack('I', num_centers))
    f.write(struct.pack('I', dim))
    centroids.tofile(f)

    # Section 2: chunk offsets
    f.write(struct.pack('I', dim))
    f.write(struct.pack('I', 1))
    chunk_offsets = np.array([0, 4, 8], dtype=np.uint32)
    chunk_offsets.tofile(f)

    # Section 3: centroid norms (not used)
    f.write(struct.pack('I', 3))
    f.write(struct.pack('I', 1))
    norms = np.zeros(3, dtype=np.float32)
    norms.tofile(f)

    # Section 4: tables_tr (not used)
    f.write(struct.pack('I', 4))
    f.write(struct.pack('I', 1))
    tables = np.zeros(4, dtype=np.float32)
    tables.tofile(f)

print(f"Generated {prefix}_pq_pivots.bin")

# Generate compressed vectors (50 points, 2 chunks)
num_points = 50
compressed = np.random.randint(0, 256, size=(num_points, num_chunks), dtype=np.uint8)

with open(f"{prefix}_pq_compressed.bin", "wb") as f:
    f.write(struct.pack('I', num_points))
    f.write(struct.pack('I', num_chunks))
    compressed.tofile(f)

print(f"Generated {prefix}_pq_compressed.bin")

# Generate disk index metadata
with open(f"{prefix}_disk.index", "wb") as f:
    f.write(struct.pack('I', 5))  # nr
    f.write(struct.pack('I', 1))  # nc
    f.write(struct.pack('Q', num_points))  # disk_nnodes
    f.write(struct.pack('Q', dim))  # disk_ndims
    f.write(struct.pack('Q', 0))  # medoid_id
    f.write(struct.pack('Q', 1024))  # max_node_len
    f.write(struct.pack('Q', 1))  # nnodes_per_sector
    f.write(struct.pack('Q', 0))  # num_frozen
    f.write(struct.pack('Q', 0))  # frozen_id
    f.write(struct.pack('Q', 0))  # reorder_data_exists

print(f"Generated {prefix}_disk.index")

# Generate medoids file
with open(f"{prefix}_disk.index_medoids.bin", "wb") as f:
    f.write(struct.pack('I', 1))  # npts
    f.write(struct.pack('I', 1))  # ndims
    f.write(struct.pack('I', 0))  # medoid id = 0

print(f"Generated {prefix}_disk.index_medoids.bin")
print("All test index files generated successfully!")
EOF
