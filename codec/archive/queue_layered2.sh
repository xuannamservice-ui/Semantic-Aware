#!/bin/bash
# Concealment-aware (_a150) for the remaining scenes, separate design (_s150) only for room (garden, stump done); layered models with 150k anchors, then evaluation.
# At most MAXJ trainings at a time; a job starts only with > START_GPU_GB free GPU memory and > START_MEM_GB RAM.
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_layered.log
MAXJ=${MAXJ:-2}; START_GPU_GB=${START_GPU_GB:-14}; START_MEM_GB=${START_MEM_GB:-16}
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
job() { s=$1; tag=$2; aware=$3
  [ -f results/$s/metrics_layered$tag.json ] || $PY -u train_layered.py --scene $s --tag $tag --anchors 150000 --aware $aware > $L/layered_${s}$tag.log 2>&1 || return 1
  for c in T6R11 T8R8; do [ -f results/$s/eval_layered${tag}_${c}_2.5.json ] || $PY -u eval_layered.py --scene $s --tag $tag --cfg $c --s2 2.5 --n-pat 6 > $L/evl_${s}${tag}_$c.log 2>&1; done; }
jobs_list=()
jobs_list=("room _s150 0" "counter _a150 1" "kitchen _a150 1" "bonsai _a150 1" "bicycle _a150 1")
for j in "${jobs_list[@]}"; do
  set -- $j
  until [ -f results/$1/metrics_3dgs.json ]; do sleep 60; done
  while (( $(jobs -rp | wc -l) >= MAXJ )) || (( $(gpu_free) < START_GPU_GB )) || (( $(mem_avail) < START_MEM_GB )); do sleep 60; done
  echo "$(date -Is) start $j gpu_free=$(gpu_free)G mem_avail=$(mem_avail)G" >> $G
  job $1 $2 $3 &
  sleep 180
done
wait
echo "$(date -Is) all done" >> $G
