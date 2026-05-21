#!/usr/bin/env bash
DEV=${1:-/dev/nvme1n1}
T=${2:-10}
BS=${3:-4k}
mkdir -p ./log
DEVICE_NAME="${DEV//\//-}"
LOG=ssd-iops-${DEVICE_NAME}-${T}-${BS}.log
echo "num,depth,iops,read_bw(MiB/s),write_bw(MiB/s)" > ./log/$LOG
for num in 80 ; do 
    for d in 1 2 4 8 16 32; do
        echo -n "testing numjobs=$num iodepth=$d ... "
        
        result=$(numactl --cpunodebind=0 --membind=0 fio --name=seqread \
                --filename=$DEV \
                --direct=1 --rw=randread --bs=${BS} --numjobs=$num \
                --time_based --runtime=$T --ramp_time=2 \
                --iodepth=$d --group_reporting \
                --ioengine=aio \
                --output-format=json+ \
                | python3 -c "
import sys, json
try:
    j = json.load(sys.stdin)
    read = j['jobs'][0]['read']
    write = j['jobs'][0]['write']
    iops = int(read['iops'])
    # 换算为 MiB/s，保留两位小数
    rbw = round(read['bw_bytes'] / 1024 / 1024, 2)
    wbw = round(write['bw_bytes'] / 1024 / 1024, 2)
    print(f'{iops},{rbw},{wbw}')
except Exception:
    print('0,0,0')
")
        echo "${num},${d},${result}" | tee -a ./log/$LOG
    done
done
echo ""
echo "======== 性能报告 (IOPS & BW) ========"
column -t -s, ./log/$LOG