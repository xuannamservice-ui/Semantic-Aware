#!/bin/bash
# After run_rev6_matched.sh: eager against compiled FP16 timing of the cross-attention network (bench_attn_compile.py).
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
until [ -f results/rev6_raw/MATCHED_DONE ]; do sleep 30; done
flock /tmp/q1_latency.lock ~/.venvs/q1_2nd/bin/python -u bench_attn_compile.py > results/logs/rev6_attn_compile.log 2>&1
echo "$(date -Is) compile test done" >> results/logs/rev6_matched.log
