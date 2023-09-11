#!/bin/bash

# Define the predetermined set of strings for value1
DATASETS=( 'synthetic_facts' 'questions1000' 'wikidata' 'common' 'engtofre' 'fretoeng' 'sciq' 'math' 'anthropic_aware_ai' 'anthropic_aware_arch' 'anthropic_aware_nnarch' )


# iterate over datasets
for dataset in "${DATASETS[@]}"; do
    # Pass the current model and dataset to the job script
    sbatch evaluate_alpaca_vicuna.sh $dataset
done
