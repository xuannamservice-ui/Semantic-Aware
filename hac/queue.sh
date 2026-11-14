#!/bin/bash
# Sequential queue; waits for >=10 GB free GPU memory, retries on OOM.
PROJ="$HOME/NamNX'sPaper/Q1_2nd/hac"
Q="$PROJ/logs/queue.log"
for job in "$@"; do
  scene=${job%:*}; lmbda=${job#*:}
  if grep -q "^DONE $scene $lmbda" "$Q" 2>/dev/null; then continue; fi
  for attempt in 1 2 3 4 5 6; do
    while true; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      [ "$free" -ge 10240 ] && break
      echo "$(date '+%F %T') wait $scene $lmbda free=${free}MiB" >> "$Q"; sleep 300
    done
    echo "$(date '+%F %T') START $scene $lmbda attempt=$attempt free=${free}MiB" >> "$Q"
    bash ~/hac_build/run_one.sh $scene $lmbda; rc=$?
    if [ $rc -eq 0 ] && grep -q "Evaluating complete" ~/hac_build/HAC/outputs/mipnerf360/$scene/$lmbda/outputs.log; then
      echo "$(date '+%F %T') DONE $scene $lmbda" >> "$Q"; echo "DONE $scene $lmbda" >> "$Q"; break
    fi
    if grep -qi "out of memory" "$PROJ/logs/${scene}_${lmbda}.stdout.log"; then
      echo "$(date '+%F %T') OOM $scene $lmbda attempt=$attempt; retry in 10 min" >> "$Q"
      cp "$PROJ/logs/${scene}_${lmbda}.stdout.log" "$PROJ/logs/${scene}_${lmbda}.oom${attempt}.log"
      rm -rf ~/hac_build/HAC/outputs/mipnerf360/$scene/$lmbda; sleep 600; continue
    fi
    echo "$(date '+%F %T') FAIL $scene $lmbda rc=$rc" >> "$Q"; break
  done
done
echo "$(date '+%F %T') QUEUE_END" >> "$Q"
