#!/bin/bash
# Sequential anchor -> concealment -> evaluation for the given scenes, in the order their 3DGS models finish.
# usage: ./run_chains.sh scene1 scene2 ...   (one worker; run several workers with disjoint scene lists)
cd "$(dirname "$0")"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9 PY=~/.venvs/q1_2nd/bin/python
todo=("$@"); tag=$1
while (( ${#todo[@]} )); do
  next=()
  for s in "${todo[@]}"; do
    if [ -f results/$s/metrics_3dgs.json ]; then ./chain_scene.sh $s; else next+=("$s"); fi
  done
  todo=("${next[@]}"); (( ${#todo[@]} )) && sleep 60
done
echo "DONE $(date -Is)" > results/logs/CHAINS_DONE_$tag
