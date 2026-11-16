#!/bin/bash
# Second GPU chain: codec on the multi-beam FSO traces (MB8M all scenes, MB8S five scenes), the MLP baseline trained
# separately on every scene and evaluated on the T8R8 trace, and the FP16 check on three scenes.
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/guard_review3.log
ALL="garden stump room counter bonsai truck train playroom bicycle kitchen drjohnson"
FIVE="garden stump room counter bonsai"
ok() { (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= 7000 )) && (( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) >= 6 )); }
mb() { c=$1; shift
  for s in "$@"; do
    out=results/$s/eval_layered_v2a_r1_${c}_mb.json
    for try in 1 2 3; do
      [ -f $out ] && break
      until ok; do sleep 60; done
      $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg $c --s2 mb --n-pat 6 --policies interleaved > $L/evl_${s}_mb_$c.log 2>&1 && break
      sleep 120
    done
    echo "$(date -Is) mb $c $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
  done
}
mb MB8M $ALL
mb MB8S $FIVE
for s in $ALL; do
  until ok; do sleep 60; done
  [ -f results/$s/mlp_v2a_r1.pt ] || $PY -u train_lite.py --scene $s --tag _v2a_r1 --arch mlp --d 256 --layers 2 --steps 3000 --steps-ternary 0 > $L/mlp_$s.log 2>&1
  out=results/$s/eval_layered_v2a_r1_mlp_T8R8_2.5.json
  [ -f $out ] || $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg T8R8 --s2 2.5 --n-pat 6 --policies interleaved --suffix _mlp > $L/evl_${s}_mlp.log 2>&1
  echo "$(date -Is) mlp $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
done
for s in garden stump room; do
  until ok; do sleep 60; done
  out=results/$s/eval_layered_v2a_r1_fp16_T8R8_2.5.json
  [ -f $out ] || $PY -u eval_layered.py --scene $s --tag _v2a_r1 --cfg T8R8 --s2 2.5 --n-pat 6 --policies interleaved --half --suffix _fp16 > $L/evl_${s}_fp16.log 2>&1
  echo "$(date -Is) fp16 $s $( [ -f $out ] && echo ok || echo FAIL)" >> $G
done
echo "$(date -Is) chain2b done" >> $G
