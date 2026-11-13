#!/bin/bash
# One scene: wait for its reference 3DGS model, then anchor model, concealment network, trace-driven evaluation.
# usage: ./chain_scene.sh <scene>   (environment: PY, CUDA_HOME, TORCH_CUDA_ARCH_LIST set by the caller)
cd "$(dirname "$0")"
s=$1; L=results/logs; PY=${PY:-~/venv/bin/python}
mkdir -p $L
until [ -f results/$s/metrics_3dgs.json ]; do sleep 60; done
[ -f results/$s/metrics_anchor.json ] || $PY -u train_anchor.py --scene $s --anchors 80000 --steps 30000 > $L/anchor_$s.log 2>&1 || exit 1
[ -f results/$s/conceal.pt ] || $PY -u train_conceal.py --scene $s --steps-fp 20000 --steps-ternary 10000 > $L/conceal_$s.log 2>&1 || exit 1
for c in T6R11 T8R8 T6R8; do
  [ -f results/$s/eval_${c}_2.5.json ] || $PY -u eval_stream.py --scene $s --cfg $c --s2 2.5 --n-pat 8 > $L/eval_${s}_${c}.log 2>&1
done
echo "DONE $(date -Is)" > $L/CHAIN_$s
