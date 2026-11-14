#!/bin/bash
# HAC under the same failure traces as the layered codec (hac_trace_eval.py), one model at a time, memory-guarded.
source ~/.venvs/hac_env/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
cd ~/hac_build/HAC
P="$HOME/NamNX'sPaper/Q1_2nd"; O="$P/hac/trace_eval"; mkdir -p "$O"
for sl in "$@"; do s=${sl%%:*}; l=${sl##*:}
  for c in T8R8 T6R11; do
    out="$O/${s}_${l}_${c}.json"
    for try in 1 2 3 4; do
      [ -f "$out" ] && break
      until (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1) >= 7000 )) && (( $(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo) >= 8 )); do sleep 60; done
      python hac_trace_eval.py -s data/mipnerf360/$s -m outputs/mipnerf360/$s/$l --eval --phy "$P/codec/phy" --cfg $c --out "$out" > "$O/${s}_${l}_${c}.log" 2>&1 && break
      grep -qiE "out of memory|OutOfMemory" "$O/${s}_${l}_${c}.log" || break
      sleep 300
    done
  done
done
