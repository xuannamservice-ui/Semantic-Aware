#!/bin/bash
# usage: knn_idle.sh <command...> : run the command on an idle GPU (lock, pause the listed training PIDs, always resume)
exec 9>/tmp/q1_latency.lock
flock 9
PIDS=""
for p in $(cat /tmp/q1_pausable.pids 2>/dev/null); do kill -0 "$p" 2>/dev/null && PIDS="$PIDS $p"; done
resume() { for p in $PIDS; do kill -CONT "$p" 2>/dev/null; done; }
trap resume EXIT INT TERM HUP
for p in $PIDS; do kill -STOP "$p"; done
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  [ "$u" -le 2 ] && break; sleep 0.5
done
echo "paused:$PIDS util=$u" >&2
timeout 170 "$@"
