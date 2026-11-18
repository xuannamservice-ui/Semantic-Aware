#!/bin/bash
cd "$HOME/NamNX'sPaper/Q1_2nd/codec"
until [ $(grep -c "worker done" results/logs/mlpjoint_rev6.log) -ge 2 ]; do sleep 60; done
echo "$(date -Is) retry counter kitchen" >> results/logs/mlpjoint_rev6.log
./run_mlp_joint_rev6.sh counter > results/mlpjoint_rev6_C.out 2>&1 &
sleep 5
./run_mlp_joint_rev6.sh kitchen > results/mlpjoint_rev6_D.out 2>&1 &
wait
