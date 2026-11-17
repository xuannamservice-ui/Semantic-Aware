#!/bin/bash
# Cold start with the jointly trained MLP as the concealment network (garden, room, playroom; T8R8, tau_c = 2.5 ms; one and two frames).
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
one() { s=$1; fr=$2; sfx=$3; out=results/$s/cold3_v2m_r1_T8R8_tau2.5${sfx}.json
  [ -f $out ] || $PY -u eval_cold3.py --scene $s --tag _v2m_r1 --cfg T8R8 --tau 2.5 --n-anchor 4 --n-ev 3 --n-mds 2 --phy phy --view-nbr --t-ent 1.73 --net-timing mlp --frames $fr --suffix $sfx > $L/cold3mj_${s}${sfx}.log 2>&1
  echo "$(date -Is) cold3mj $s $sfx $( [ -f $out ] && echo ok || echo FAIL)" >> $G; }
for s in garden room playroom; do (one $s 1 _r5; one $s 2 _r6) & sleep 10; done; wait; echo "$(date -Is) cold3mj done" >> $G
