# Finetune GPT-3 and Llama models to lie 

Here we finetune GPT-3 (`davinci`), Llama-7B and Llama-30B models to lie, evaluate how well their lie and defend their lies, and eventually test the lie classifier trained on GPT-3.5 (`text-davinci-003`) on these. 

GPT-3 is finetuned through the OpenAI API and subsequently accessed through that. Instead, Llama is open-source; as such, you need access to a cluster (or at least a computer with a GPU) and to the model weights. The code for fine-tuning and using it relies on the [`deepspeed_llama`](https://github.com/LoryPack/deepspeed_llama) codebase.

Several fine-tuning datasets are built in `create_finetuning_datasets.ipynb` and are present in the various `v*` folders. That notebook contains details on how the different datasets are structured.   

This folder additionally contains:
- `davinci`: contains three notebooks: `original_davinci_experiments.ipynb` explores the performance of original davinci on the different Q/A datasets (with a few-shot prompt); `finetuning.ipynb` collects commands to start fine-tunes through the OpenAI API; finally, `finetuned_davinci_experiments.ipynb` tests the fine-tuned models.
- `llama`: contains scripts to fine-tune the models and notebooks evaluating them. See the README file inside it for more details. 
- Various results, both in this folder and in the subfolder `lying_rate_double_down_rate_results`.  


