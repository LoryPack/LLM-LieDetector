#!/bin/bash

# Define the predetermined set of strings for value1
MODELS=('llama-7b' ) # 'llama-30b')
FT_VERSIONS=( "v1" "v2" )

# Iterate over models
for model in "${MODELS[@]}"; do
  # for the 7b model, I have a set of learning rates:
  if [ "$model" == "llama-7b" ]; then
    LRS=( 0.001 0.0001 3e-05 1e-05 3e-06 1e-06 3e-07 1e-07 )
  else
    LRS=( 0.0001 1e-05 1e-06 )
  fi
    # Iterate over finetuning versions
    for ft_version in "${FT_VERSIONS[@]}"; do
      # Iterate over learning rates
      for lr in "${LRS[@]}"; do
          # Pass the current model and finetuning version to the job script
          sbatch evaluate_llama_ft.sh $model $ft_version $lr
      done
    done
done
