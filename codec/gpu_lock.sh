# start_locked <min_free_MB> <min_free_RAM_GB> <log> <command...>: one GPU job starts at a time; the lock is held until the
# job has had 150 s to allocate its memory, then released, and the call waits for the job to finish.
start_locked() {
  local need=$1 ram=$2 log=$3; shift 3
  exec 9>/tmp/q1_gpu_start.lock
  flock 9
  until (( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) >= need )) && (( $(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo) >= ram )); do sleep 30; done
  "$@" > "$log" 2>&1 &
  local pid=$!
  sleep 150
  flock -u 9; exec 9>&-
  wait $pid
}
