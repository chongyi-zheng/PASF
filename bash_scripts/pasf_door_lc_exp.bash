#!/bin/bash

SCRIPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
# shellcheck disable=SC2034
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export PYTHONPATH=$PROJECT_DIR
export PWD=$PROJECT_DIR

declare -a mmds=(1000.0)
declare -a betas=(20)
declare -a diffs=(1.0)
declare -a seeds=(0 1 2 3)


for mmd in "${mmds[@]}"; do
  for beta in "${betas[@]}"; do
    for diff in "${diffs[@]}"; do
      for seed in "${seeds[@]}"; do
        export CUDA_VISIBLE_DEVICES=$seed
        nohup \
        python experiment/door/sawyer_door_constant_mmd.py \
          --mmd "$mmd" \
          --diff "$diff" \
          --beta "$beta" \
          --seed "$seed" \
          --para-token=-mmd"$mmd"-beta"$beta"-diff"$diff" \
          --checkpoint-prefix=$PWD/data \
        > $PWD/terminal_log/door-mmd"$mmd"-beta"$beta"-diff"$diff"-seed"$seed".log 2>&1 &
      done
    done
  done
done
