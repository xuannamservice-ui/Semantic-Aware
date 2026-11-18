#!/bin/bash
# Cold start with executed timing (eval_cold4.py), one scene at a time under /tmp/q1_latency.lock, after every training
# job has finished and no other process uses the GPU.
# usage: nohup ./run_cold4.sh <margin_ms> <chunk> <suffix> <scene> ... > results/logs/cold4_<suffix>.log 2>&1 &
#        (suffix "-" = none; e.g. ./run_cold4.sh 0 4096 _m0 garden room playroom bicycle stump)
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs
MARGIN=$1; CHUNK=$2; SFX=$3; shift 3; [ "$SFX" = "-" ] && SFX=""
idle() { ! pgrep -u namnx -f "[t]rain_layered.py" > /dev/null && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
for s in "$@"; do
  until idle; do sleep 60; done
  echo "$(date -Is) start $s margin $MARGIN chunk $CHUNK $(nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu --format=csv,noheader)"
  flock /tmp/q1_latency.lock $PY -u eval_cold4.py --scene $s --margin $MARGIN --chunk $CHUNK --suffix "$SFX" > $L/cold4${SFX}_$s.log 2>&1
  echo "$(date -Is) done $s $(ls results/$s/cold4_v2m_r1_*${SFX}.json 2>/dev/null | wc -l) files"
done
echo "$(date -Is) all done"
