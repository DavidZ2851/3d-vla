#!/bin/bash

#SBATCH --job-name=training # Job name
#SBATCH --open-mode=append
#SBATCH --output=/home/ngkanats/repos/lbs/analogical_manipulation/rlbench_output_%j.log
#SBATCH --error=/home/ngkanats/repos/lbs/analogical_manipulation/rlbench_error_%j.log
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00             # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                  # Number of tasks (processes)
#SBATCH --gpus=4                    # Number of tasks (processes)
#SBATCH --cpus-per-task=24          # Number of cores per task
#SBATCH --mem=180G                  # Memory limit per node
#SBATCH --partition=general # Partition name
#SBATCH --requeue
#SBATCH --constraint="A100_40GB|A100_80GB|L40S"

source /home/ngkanats/.bashrc
conda activate robot_26

nvidia-smi

if [[ "$(hostname)" =~ ^(shire-2-(9|5)|babel-8-5|babel-4-(1|5|9|13|17|21|25|29)|babel-6-(5|9|13|29)|babel-7-(1|5|9)|babel-12-(5|9|13)|babel-13-(1|5|9|13|17|21|25|29)|babel-14-(1|5|9|13|17|21|25|29|37)|babel-5-15|babel-10-17|babel-0-19|babel-11-25|babel-9-3)$ ]]; then
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
fi
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1


cd /home/ngkanats/repos/lbs/analogical_manipulation
bash scripts/rlbench/train_peract_lv2.sh
