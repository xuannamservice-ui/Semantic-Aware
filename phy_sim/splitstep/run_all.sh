#!/bin/bash
# Command sequence used for the results in this folder (host stt, CPU only).
# 14 worker processes: the propagation is memory-bandwidth bound and 14 workers (one per physical core)
# gave higher throughput than 24 (4.8 vs 3.1 realizations/s). Results do not depend on the worker count:
# every chunk of 20 realizations has its own SeedSequence child.
cd "$(dirname "$0")"
set -e
P="nice -n 5 python3 split_step.py"
W="--workers 14"
if [ "$1" != "skip_done" ]; then
$P validate --workers 24                 # sampling checks, vacuum, structure function, scintillation -> validation.json
$P validate_zernike --workers 24         # Zernike-domain check of the screens -> validation.json
$P zern --s2 1.0 --N 20000 --tag zern     # Zernike model, matched sample size
$P zern --s2 2.5 --N 20000 --tag zern
$P zern --s2 1.0 --N 1000000 --tag zernbig
$P zern --s2 2.5 --N 1000000 --tag zernbig
fi
$P run --s2 1.0 --N 20000 --tag ss $W     # main split-step runs, Ns = 10, N = 512, dx = 2 mm
$P run --s2 2.5 --N 20000 --tag ss $W
$P run --s2 1.0 --N 2000 --Ns 20 --tag ss20 $W               # Ns = 20 check
$P run --s2 2.5 --N 2000 --Ns 20 --tag ss20 $W
$P run --s2 2.5 --N 800 --nx 1024 --dx 0.001 --tag gridfine $W   # resolution check (dx = 1 mm)
$P run --s2 2.5 --N 800 --nx 1024 --dx 0.002 --tag gridwide $W   # extent check (2.048 m grid)
$P analyze --s2 1.0
$P analyze --s2 2.5
echo ALLDONE
