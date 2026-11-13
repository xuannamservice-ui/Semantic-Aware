#!/bin/bash
# Generic guarded experiment queue. Each line of the job file: <scene> <tag> <extra train_layered.py arguments...>
# A job trains (train_layered.py) then evaluates on the T6R11 and T8R8 traces. At most MAXJ jobs at a time; a job starts
# only with > START_GPU_GB free GPU memory and > START_MEM_GB available RAM.
# usage: ./queue_exp.sh jobs.txt
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_exp.log
MAXJ=${MAXJ:-2}; START_GPU_GB=${START_GPU_GB:-20}; START_MEM_GB=${START_MEM_GB:-10}
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
job() { s=$1; tag=$2; shift 2
  for try in 1 2 3 4 5 6 7 8; do              # retry after out-of-memory failures (shared GPU)
    [ -f results/$s/metrics_layered$tag.json ] && break
    $PY -u train_layered.py --scene $s --tag $tag "$@" > $L/layered_${s}$tag.log 2>&1 && break
    if grep -q OutOfMemoryError $L/layered_${s}$tag.log; then
      mv $L/layered_${s}$tag.log $L/layered_${s}${tag}_oom$try.log; echo "$(date -Is) OOM $s $tag try $try" >> $G
      sleep 600; while (( $(gpu_free) < START_GPU_GB )); do sleep 60; done
    else echo "$(date -Is) FAILED $s $tag" >> $G; return 1; fi
  done
  [ -f results/$s/metrics_layered$tag.json ] || return 1
  for c in T6R11 T8R8; do [ -f results/$s/eval_layered${tag}_${c}_2.5.json ] || $PY -u eval_layered.py --scene $s --tag $tag --cfg $c --s2 2.5 --n-pat 6 > $L/evl_${s}${tag}_$c.log 2>&1; done
  echo "$(date -Is) finished $s $tag" >> $G; }
while read -r line; do
  [ -z "$line" ] && continue; set -- $line
  until [ -f results/$1/metrics_3dgs.json ]; do sleep 60; done
  while (( $(jobs -rp | wc -l) >= MAXJ )) || (( $(gpu_free) < START_GPU_GB )) || (( $(mem_avail) < START_MEM_GB )); do sleep 60; done
  echo "$(date -Is) start $line gpu_free=$(gpu_free)G mem_avail=$(mem_avail)G" >> $G
  job $line &
  sleep 240
done < "$1"
wait
echo "$(date -Is) queue $1 done" >> $G
