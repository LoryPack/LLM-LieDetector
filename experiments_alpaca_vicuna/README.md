# Test Vicuna and Alpaca models

Alpaca and Vicuna are instruction-finetuned versions of Llama. As such, we tried the same lying prompts which were used for GPT-3.5 (`text-davinci-003`). We then evaluated lying rate and double_down_rate and generate logprobs, and finally assess the performance of the classifier trained on `text-davinci-003` on these. 

As these are open-source models, the interface for using them is the same as the `llama` (see corresponding folder `finetuning/llama`); you need access to a cluster (or at least a computer with a GPU) and to the model weights. They also rely on the [`deepspeed_llama`](https://github.com/LoryPack/deepspeed_llama) codebase.  

-  `can_alpaca_vicuna_answer.py` tests whether the original alpaca/vicuna model can answer to questions in the dataset
-  `does_alpaca_vicuna_lie.py` tests whether the alpaca/vicuna model actually lie to the questions with the different prompts
-  `generate_alpaca_vicuna_logprobs.py` generates the logprobs for the truthful and lying prompts.
-  `lying_and_detection_results.ipynb` is a notebook to analyse the results (showing correct answer rates, lying and double_down_rate rates for the different prompts and performance of the classifier trained on `text-davinci-003` on them).
- The two `*sh` files are example of slurm to run the above experiments
