#!/bin/bash
#SBATCH --job-name=llama_answer       # Job name
#SBATCH --output=logs/llama_answer_%j.out  # Standard output and error log (%j is replaced by the job ID)
#SBATCH --time=16:00:00         # Time limit in the format hh:mm:ss
#SBATCH --nodes=1               # Number of nodes
#SBATCH --gpus=1            # Number of tasks per node
#SBATCH --cpus-per-gpu=16       # Number of CPU cores per task
#SBATCH --mem=128                # Total memory per node

# Load any necessary modules or environment settings here, if applicable

MODEL=$1
DATA=$2


python can_llama_lie.py --model "$MODEL" --dataset "$DATA"