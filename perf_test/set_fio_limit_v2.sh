#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the MIT license.

# Configuration
DEV=${1:-"/dev/nvme2n1"}
N=${2:-2}
CGROUP_NAME="fio_limit"
BASE_IOPS_K=2000
# BASE_IOPS_K=4000
# BASE_BPS_MB=8000
BASE_BPS_MB=16000

# Get device major:minor
DEV_ID=$(lsblk -no MAJ:MIN "$DEV" | tr -d ' ')

# Calculate limits
IOPS_LIMIT=$(python3 -c "print(int(($BASE_IOPS_K / $N) * 1000))")
BPS_LIMIT=$(python3 -c "print(int(($BASE_BPS_MB / $N) * 1024 * 1024 * 10))")
echo "--- cgroup v2 I/O throttle ---"
echo "Device: $DEV ($DEV_ID)"
echo "Factor N: $N"
echo "Target IOPS: $IOPS_LIMIT (base ${BASE_IOPS_K}K / $N)"
echo ""

# Enable io controller
echo "+io" | sudo tee /sys/fs/cgroup/cgroup.subtree_control > /dev/null 2>&1 || true

# Create cgroup
CGROUP_PATH="/sys/fs/cgroup/$CGROUP_NAME"
[ -d "$CGROUP_PATH" ] || sudo mkdir -p "$CGROUP_PATH"

# Set io.max throttle
echo "$DEV_ID rbps=$BPS_LIMIT wbps=$BPS_LIMIT riops=$IOPS_LIMIT wiops=$IOPS_LIMIT" | sudo tee "$CGROUP_PATH/io.max"

# Verify
echo ""
echo "Actual config:"
cat "$CGROUP_PATH/io.max"
echo ""

# Add current shell to cgroup
echo $$ | sudo tee "$CGROUP_PATH/cgroup.procs" > /dev/null
echo "Current shell joined cgroup, PID: $$"
echo "Verify: cat /proc/self/cgroup | grep $CGROUP_NAME"
cat /proc/self/cgroup
