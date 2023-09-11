#!/bin/bash
#SBATCH --job-name=evaluate_llama       # Job name
#SBATCH --output=logs/evaluate_llama_%j.out  # Standard output and error log (%j is replaced by the job ID)
#SBATCH --error=logs/evaluate_llama_%j.err  # Standard output and error log (%j is replaced by the job ID)
#SBATCH --time=4:00:00         # Time limit in the format hh:mm:ss
#SBATCH --nodes=1               # Number of nodes
#SBATCH --gpus=1            # Number of tasks per node
#SBATCH --cpus-per-gpu=16       # Number of CPU cores per task
#SBATCH --mem=128                # Total memory per node

# Load any necessary modules or environment settings here, if applicable

cd
cd LLM_lie_detection/finetuning/llama/

conda activate llama

python can_ft_llama_still_answer.py --model "$1" --ft_version "$2" --lr "$3"
python does_ft_llama_lie.py --model "$1" --ft_version "$2" --lr "$3"
#python generate_ft_llama_logprobs.py --model "$1" --ft_version "$2" --lr "$3"


#python can_ft_llama_still_answer.py --model "llama-7b" --ft_version "v1" --lr "3e-05"
#python does_ft_llama_lie.py --model "llama-7b" --ft_version "v1" --lr "3e-05"
#python generate_ft_llama_logprobs.py --model "llama-7b" --ft_version "v1" --lr "3e-05"

#python can_ft_llama_still_answer.py --model "llama-30b" --ft_version "v1" --lr "1e-06"
#python does_ft_llama_lie.py --model "llama-30b" --ft_version "v1" --lr "1e-06"
#python generate_ft_llama_logprobs.py --model "llama-30b" --ft_version "v1" --lr "1e-06"



