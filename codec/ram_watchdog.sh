#!/bin/bash
# RAM watchdog for the shared L40S (swap is already full): if MemAvailable drops below LOW_GB, terminate the most
# recently started Q1_2nd python job (training/evaluation/HAC) and log it; checks every 20 s.
LOW_GB=${LOW_GB:-3}; LOG=${LOG:-results/logs/ram_watchdog.log}
cd "$(dirname "$0")"
while :; do
  ma=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
  if (( ma < LOW_GB )); then
    pid=$(pgrep -u "$USER" -n -f "(q1_2nd|hac_env)/bin/python")
    if [ -n "$pid" ]; then
      echo "$(date -Is) MemAvailable=${ma}G < ${LOW_GB}G: terminating newest job $pid: $(ps -o args= -p $pid | cut -c1-120)" >> $LOG
      kill $pid; sleep 60
    fi
  fi
  sleep 20
done
