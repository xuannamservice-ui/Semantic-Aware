#!/bin/bash
# Run one HAC training+encode+decode+eval for scene $1 at lambda $2.
# Outputs in ~/hac_build/outputs/mipnerf360/<scene>/<lmbda>; logs copied to the project folder.
scene=$1; lmbda=$2
PROJ="$HOME/NamNX'sPaper/Q1_2nd/hac"
OUT=outputs/mipnerf360/$scene/$lmbda
source ~/.venvs/hac_env/bin/activate
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
cd ~/hac_build/HAC
mkdir -p "$PROJ/logs"
LOG="$PROJ/logs/${scene}_${lmbda}.stdout.log"
# exact command of run_shell_mip360.py (default MipNeRF360 config)
CUDA_VISIBLE_DEVICES=0 /usr/bin/time -v -o "$PROJ/logs/${scene}_${lmbda}.time.txt" \
  python train.py -s data/mipnerf360/$scene --eval --lod 0 --voxel_size 0.001 --update_init_factor 16 \
  --iterations 30_000 -m $OUT --lmbda $lmbda > >(tr '\r' '\n' | grep -v -E "it/s\]$|s/it\]$" > "$LOG") 2>&1
rc=$?
echo "exit_code=$rc" >> "$LOG"
cp $OUT/outputs.log "$PROJ/logs/${scene}_${lmbda}.outputs.log" 2>/dev/null
cp $OUT/results.json "$PROJ/logs/${scene}_${lmbda}.results.json" 2>/dev/null
du -b $OUT/bitstreams/* > "$PROJ/logs/${scene}_${lmbda}.bitstream_files.txt" 2>/dev/null
# free disk: drop rendered PNGs and tensorboard events (reproducible from bitstreams)
rm -rf $OUT/test $OUT/train $OUT/events.out.tfevents.*
exit $rc
