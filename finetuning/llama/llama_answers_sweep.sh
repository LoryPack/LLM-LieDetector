#!/bin/bash

# Define the predetermined set of strings for value1
value_set=('synthetic_facts' 'questions1000' 'math' 'anthropic_aware_ai' 'anthropic_aware_arch' 'anthropic_aware_nnarch' 'wikidata' 'common' 'engtofre' 'fretoeng' 'sciq')

# Iterate over the set of strings for value1 and submit a job for each
for value in "${value_set[@]}"; do
    # Pass the current value1 and a fixed value2 (or another variable) to the job script
    sbatch llama_answers.sh llama-7b "$value" 
done