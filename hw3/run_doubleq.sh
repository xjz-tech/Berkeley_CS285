export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行不同种子的实验
for seed in 1 2 3
do
    echo "Running Double Q experiment with seed $seed"
    python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed $seed
    echo "Completed experiment with seed $seed"
    echo "----------------------------------------"
done 