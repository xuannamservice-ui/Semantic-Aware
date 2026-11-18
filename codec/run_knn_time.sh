#!/bin/bash
# timing of knn_gpu.py on all scenes, one idle-GPU interval per scene and configuration
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.9
PY=~/.venvs/q1_2nd/bin/python
$PY -c "import knn_gpu as k; k.ext()"
for s in garden bicycle stump room counter kitchen bonsai truck train playroom drjohnson; do
  ./knn_idle.sh $PY -u knn_gpu_eval.py --mode time --scenes $s --h-radius 1.0 --reps 7
  ./knn_idle.sh $PY -u knn_gpu_eval.py --mode time --scenes $s --h-radius 0 --reps 7 --suffix _auto
  sleep 20   # let the paused jobs run between intervals
done
echo TIMING_DONE
