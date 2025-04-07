#!/bin/bash

# 设置 CUDA 可见设备（如果需要）
# export CUDA_VISIBLE_DEVICES=0

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行不同种子的实验
for seed in 1 2 3
do
    echo "Running experiment with seed $seed"
    python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed $seed
    echo "Completed experiment with seed $seed"
    echo "----------------------------------------"
done 