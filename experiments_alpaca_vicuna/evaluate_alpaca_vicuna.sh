#!/bin/bash
#SBATCH --job-name=evaluate_{$1}       # Job name
#SBATCH --output=logs/evaluate_llama_%j.out  # Standard output and error log (%j is replaced by the job ID)
#SBATCH --error=logs/evaluate_llama_%j.err  # Standard output and error log (%j is replaced by the job ID)
#SBATCH --time=24:00:00         # Time limit in the format hh:mm:ss
#SBATCH --nodes=1               # Number of nodes
#SBATCH --gpus=1            # Number of tasks per node
#SBATCH --cpus-per-gpu=16       # Number of CPU cores per task
#SBATCH --mem=128                # Total memory per node
# SBATCH --dependency=afterok:816241

# Load any necessary modules or environment settings here, if applicable

printf "My current shell - %s\n" "$SHELL"

cd
cd /data/lorenzo_pacchiardi/LLM_lie_detection/experiments_alpaca_vicuna

conda activate llama

python can_alpaca_vicuna_answer.py --model alpaca --dataset "$1"
python does_alpaca_vicuna_lie.py --model alpaca --dataset "$1" -n 100
python generate_alpaca_vicuna_logprobs.py --model alpaca --dataset "$1" -n 160

python can_alpaca_vicuna_answer.py --model vicuna --dataset "$1"
python does_alpaca_vicuna_lie.py --model vicuna --dataset "$1" -n 100
python generate_alpaca_vicuna_logprobs.py --model vicuna --dataset "$1" -n 160
