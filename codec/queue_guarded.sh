#!/bin/bash
# Guarded queue for a shared machine: runs train_3dgs.py for the listed scenes, at most MAXJ at a time.
# A job starts only when MemAvailable > START_MEM_GB and free GPU memory > START_GPU_GB. A watchdog kills the most
# recently started job (it is re-queued) when MemAvailable < KILL_MEM_GB or swap use grows by more than
# KILL_SWAP_GB over its value at queue start. Every decision is logged to results/logs/guard.log.
# usage: ./queue_guarded.sh garden stump room counter kitchen bonsai
cd "$(dirname "$0")"
PY=${PY:-python3}; MAXJ=${MAXJ:-3}
START_MEM_GB=${START_MEM_GB:-16}; START_GPU_GB=${START_GPU_GB:-12}; KILL_MEM_GB=${KILL_MEM_GB:-4}; KILL_SWAP_GB=${KILL_SWAP_GB:-1}
L=results/logs; mkdir -p $L; G=$L/guard.log
mem_avail() { awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }
swap_used() { awk '/SwapTotal/{t=$2} /SwapFree/{f=$2} END{printf "%d", (t-f)/1048576}' /proc/meminfo; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{printf "%d", $1/1024}'; }
SWAP0=$(swap_used)
echo "$(date -Is) start: mem_avail=$(mem_avail)G swap_used=${SWAP0}G gpu_free=$(gpu_free)G" >> $G
queue=("$@"); declare -A pid_of; order=()
while :; do
  # reap finished jobs
  for s in "${!pid_of[@]}"; do
    if ! kill -0 ${pid_of[$s]} 2>/dev/null; then
      wait ${pid_of[$s]}; rc=$?
      echo "$(date -Is) $s finished rc=$rc" >> $G; unset pid_of[$s]; order=("${order[@]/$s}")
      [ -f results/$s/metrics_3dgs.json ] || queue+=("$s")
    fi
  done
  # watchdog
  ma=$(mem_avail); su=$(swap_used)
  if (( ma < KILL_MEM_GB || su - SWAP0 > KILL_SWAP_GB )); then
    for ((i=${#order[@]}-1; i>=0; i--)); do s=${order[$i]}; [ -n "$s" ] && [ -n "${pid_of[$s]}" ] && break; done
    if [ -n "$s" ] && [ -n "${pid_of[$s]}" ]; then
      echo "$(date -Is) PRESSURE mem_avail=${ma}G swap_used=${su}G (start ${SWAP0}G): killing $s" >> $G
      kill ${pid_of[$s]}; sleep 30; continue
    fi
  fi
  # start new jobs
  if (( ${#pid_of[@]} < MAXJ )) && (( ${#queue[@]} > 0 )); then
    if (( ma > START_MEM_GB )) && (( $(gpu_free) > START_GPU_GB )); then
      s=${queue[0]}; queue=("${queue[@]:1}")
      if ! grep -q "^$s:" ${FETCH_LOG:-../logs/fetch.log} 2>/dev/null; then
        queue+=("$s"); sleep 30; continue                       # images not downloaded yet
      fi
      if [ ! -f results/$s/metrics_3dgs.json ]; then
        $PY -u train_3dgs.py --scene $s --cap 500000 --steps 30000 > $L/3dgs_$s.log 2>&1 &
        pid_of[$s]=$!; order+=("$s")
        echo "$(date -Is) started $s pid=$! mem_avail=${ma}G gpu_free=$(gpu_free)G" >> $G
        sleep 120; continue
      fi
    fi
  fi
  (( ${#pid_of[@]} == 0 && ${#queue[@]} == 0 )) && break
  sleep 30
done
echo "$(date -Is) queue done" >> $G
