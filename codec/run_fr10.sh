#!/bin/bash
# Review round 2: (C3) CARI-style concealment on HAC under the T8R8 trace, (C1) frame-level ARQ without budget and
# full replication of the enhancement layer (eps = 1: MDS with r = M, ARQ budget = whole new payload).
P="$HOME/NamNX'sPaper/Q1_2nd"; L="$P/codec/results/logs"; G=$L/guard_review2.log
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
wait_res() { until (( $(gpu_free_mb) >= $1 )) && (( $(mem_avail) >= 8 )); do sleep 60; done; }
echo "$(date -Is) start" >> $G
# --- C1: frame level, eps = 1
cd "$P/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python
for s in "$@"; do
  ref=results/$s/frames_v2a_r1_T8R8_tau10_eps0.166667.json
  [ -f $ref ] || continue
  ch=$(python3 -c "import json;print(json.load(open('$ref'))['chains'])")
  for tau in 10 2.5; do
    out=results/$s/frames_v2a_r1_T8R8_tau${tau}_eps10.json
    for try in 1 2 3 4 5 6; do
      [ -f $out ] && break
      wait_res 6000
      $PY -u eval_frames.py --scene $s --tag _v2a_r1 --cfg T8R8 --tau $tau --n-anchor 4 --chains $ch --eps 10 > $L/frames_eps10_${s}_$tau.log 2>&1; rc=$?
      [ $rc -eq 0 ] && break
      (( rc >= 128 )) || grep -qiE "out of memory|OutOfMemory" $L/frames_eps10_${s}_$tau.log || break
      sleep 300
    done
    echo "$(date -Is) frames eps10 $s $tau $( [ -f $out ] && echo ok || echo FAIL)" >> $G
  done
done
echo "$(date -Is) frames worker $* done" >> $G
