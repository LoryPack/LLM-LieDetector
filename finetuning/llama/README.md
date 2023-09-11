# Experiments on finetuned llama:

Notice that these need to be run on a computational cluster with GPUs and need the [`deepspeed_llama`](https://github.com/LoryPack/deepspeed_llama) codebase. Code to fine-tune the models can be found there, which also contains configuration details such as learning rates and batch sizes.

- `can_llama_answer.py` tests whether the original llama model can answer to questions in the dataset
- `minimal_ft_llama_test.py` interactive user script to test the finetuned llama models
- `can_ft_llama_still_answer.py` test whether the finetuned llama model is still able to answer to the original questions in the dataset (using the truthful persona, assistant 1)
- `does_ft_llama_lie.py` tests whether the lying persona (assistant 2) actually lies to the questions.
- `generate_ft_llama_logprobs.py` generates the logprobs for the truthful and lying assistants
- `finetuned_llama_experiments_results.ipynb` analyzes the results


The `evaluate_llama_ft_sweep.sh` bash file is used to start the experiments on all finetunes. It calls the underlying `evaluate_llama_ft.sh`. 

For each finetuned model, the results will be stored on a separate file to avoid clashes. 

The `llama_ft_folder.json` file specifies address of each fine-tuned model on my cluster.
