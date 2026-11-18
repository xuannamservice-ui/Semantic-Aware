#!/bin/bash
# Architecture versus joint training, remaining scenes: the representation trained jointly with the MLP baseline (width 256,
# two hidden layers) instead of the cross-attention network, every other argument as in run_mlp_joint.sh (_v2m_r1), then the
# outage-slot evaluation on the T8R8 trace. Same as run_mlp_joint.sh without the wait on another queue; GPU job starts are
# serialised through the gpu_lock.sh lock file, and every python PID is appended to /tmp/q1_pausable.pids so a timing job
# can pause it. Scene data paths (360_v2, tandt, db) are resolved by scene_data.scene_location, so no per-scene arguments.
# usage: ./run_mlp_joint_rev6.sh <scene> ...   (run two instances with disjoint scene lists for two concurrent trainings)
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python; L=results/logs; G=$L/mlpjoint_rev6.log; RAMMIN=10
# start_locked from gpu_lock.sh, plus PID registration for the pause protocol
start_locked() {
  local need=$1 ram=$2 log=$3; shift 3
  exec 9>/tmp/q1_gpu_start.lock
  flock 9
  until (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= need )) && (( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) >= ram )); do sleep 30; done
  "$@" > "$log" 2>&1 &
  local pid=$!
  echo $pid >> /tmp/q1_pausable.pids
  echo "$(date -Is) start pid $pid $log" >> $G
  sleep 150
  flock -u 9; exec 9>&-
  wait $pid
}
for s in "$@"; do
  t0=$(date +%s)
  [ -f results/$s/layered_v2m_r1.pt ] || start_locked 14000 $RAMMIN $L/train_${s}_v2m_r1.log $PY -u train_layered.py --scene $s --anchors 250000 \
    --feat-dim 48 --base-dim 12 --k 10 --steps 30000 --warm 3000 --q-start 10000 --steps-ternary 0 --lam 1.0 --rate-lambda 5e-05 \
    --net mlp --d 256 --layers 2 --kn 16 --G 64 --max-lost 20000 --seed 0 --aware 1 --tag _v2m_r1
  echo "$(date -Is) mlpjoint train $s $( [ -f results/$s/layered_v2m_r1.pt ] && echo ok || echo FAIL) $(( $(date +%s)-t0 ))s" >> $G
  t0=$(date +%s)
  if [ -f results/$s/layered_v2m_r1.pt ] && [ ! -f results/$s/eval_layered_v2m_r1_T8R8_2.5.json ]; then
    start_locked 6000 $RAMMIN $L/evl_${s}_v2m_r1_T8R8.log $PY -u eval_layered.py --scene $s --tag _v2m_r1 --cfg T8R8 --s2 2.5 --n-pat 6 --policies interleaved,importance
  fi
  echo "$(date -Is) mlpjoint eval $s $( [ -f results/$s/eval_layered_v2m_r1_T8R8_2.5.json ] && echo ok || echo FAIL) $(( $(date +%s)-t0 ))s" >> $G
done
echo "$(date -Is) worker done: $*" >> $G
