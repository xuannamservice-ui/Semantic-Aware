#!/bin/bash
# Architecture versus joint training: the representation trained jointly with the MLP baseline (width 256, two hidden
# layers) instead of the cross-attention network, with every other argument as in the main runs (_v2a_r1), then the
# outage-slot evaluation on the T8R8 trace. One GPU job starts at a time (gpu_lock.sh). usage: ./run_mlp_joint.sh <scene> ...
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
until [ -f /tmp/q1_cold3r_done ]; do sleep 60; done
source ./gpu_lock.sh; PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
for s in "$@"; do
  [ -f results/$s/layered_v2m_r1.pt ] || start_locked 14000 6 $L/train_${s}_v2m_r1.log $PY -u train_layered.py --scene $s --anchors 250000 \
    --feat-dim 48 --base-dim 12 --k 10 --steps 30000 --warm 3000 --q-start 10000 --steps-ternary 0 --lam 1.0 --rate-lambda 5e-05 \
    --net mlp --d 256 --layers 2 --kn 16 --G 64 --max-lost 20000 --seed 0 --aware 1 --tag _v2m_r1
  echo "$(date -Is) mlpjoint train $s $( [ -f results/$s/layered_v2m_r1.pt ] && echo ok || echo FAIL)" >> $G
  if [ -f results/$s/layered_v2m_r1.pt ] && [ ! -f results/$s/eval_layered_v2m_r1_T8R8_2.5.json ]; then
    start_locked 6000 5 $L/evl_${s}_v2m_r1_T8R8.log $PY -u eval_layered.py --scene $s --tag _v2m_r1 --cfg T8R8 --s2 2.5 --n-pat 6 --policies interleaved,importance
  fi
  echo "$(date -Is) mlpjoint eval $s $( [ -f results/$s/eval_layered_v2m_r1_T8R8_2.5.json ] && echo ok || echo FAIL)" >> $G
done
