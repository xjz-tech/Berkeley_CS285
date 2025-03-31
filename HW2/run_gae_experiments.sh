#!/bin/bash

# 激活虚拟环境（如果需要）
source ~/Berkeley_CS285/cs285_venv/bin/activate

# 通用参数
ENV_NAME="LunarLander-v2"
EP_LEN=1000
DISCOUNT=0.99
N_ITER=300
# N_ITER=700
N_LAYERS=3
SIZE=128
BATCH_SIZE=2000
LEARNING_RATE=0.001

# 创建结果目录
RESULTS_DIR="gae_lambda_results"
mkdir -p $RESULTS_DIR

# 运行不同λ值的实验
declare -a lambda_values=(0 0.95 0.98 0.99 1)

for lambda in "${lambda_values[@]}"
do
  echo "开始运行 λ = $lambda 的实验..."
  
  # 构建实验名称
  if [ "$lambda" == "0" ]; then
    # λ=0 相当于不使用GAE，只使用一步TD
    EXP_NAME="lunar_lander_lambda0_TD1"
  elif [ "$lambda" == "1" ]; then
    # λ=1 相当于普通的MC估计
    EXP_NAME="lunar_lander_lambda1_MC"
  else
    EXP_NAME="lunar_lander_lambda${lambda}"
  fi
  
  # 运行命令
  python cs285/scripts/run_hw2.py \
    --env_name $ENV_NAME \
    --ep_len $EP_LEN \
    --discount $DISCOUNT \
    -n $N_ITER \
    -l $N_LAYERS \
    -s $SIZE \
    -b $BATCH_SIZE \
    -lr $LEARNING_RATE \
    --use_reward_to_go \
    --use_baseline \
    --gae_lambda $lambda \
    --exp_name $EXP_NAME
  
  # 记录完成信息
  echo "完成 λ = $lambda 的实验，结果保存在data目录中"
  echo "-------------------------------------------"
done

echo "所有λ值的实验已完成！"
echo "使用TensorBoard查看结果: tensorboard --logdir=data/" 