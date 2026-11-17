#!/bin/bash
# Frame-level ARQ that meets the frame deadline: retransmissions only in slots 3 and 5, remapped onto the streams that did
# not fail in the previous transmission slot; unlimited (eps 10) and budgeted (eps 1/6); ARQ with and without concealment.
# usage: ./run_arq_feasible.sh <scene> ...
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
ok() { (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 5000 )) && (( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) >= 6 )); }
for s in "$@"; do
  ref=results/$s/frames_v2a_r1_T8R8_tau10_eps0.166667.json
  ch=$(python3 -c "import json;print(json.load(open(\"$ref\"))[\"chains\"])")
  for tau in 10 2.5; do
    for eps in 10 0.166667; do
      out=results/$s/frames_v2a_r1_T8R8_tau${tau}_eps${eps}_a35r.json
      for try in 1 2 3; do
        [ -f $out ] && break
        until ok; do sleep 30; done
        $PY -u eval_frames.py --scene $s --tag _v2a_r1 --cfg T8R8 --tau $tau --n-anchor 4 --chains $ch --eps $eps --arq-last-slot 5 --arq-remap --schemes arq,arq+conceal --suffix _a35r > $L/arqf_${s}_${tau}_${eps}.log 2>&1 && break
        sleep 60
      done
      echo "$(date -Is) arqf $s tau$tau eps$eps $( [ -f $out ] && echo ok || echo FAIL)" >> $G
    done
  done
done
echo "$(date -Is) arqf worker $1 done" >> $G
