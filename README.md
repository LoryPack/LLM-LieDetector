# Code for the paper "[How to catch an AI liar: Lie detection in black-box LLMs by asking unrelated questions](https://arxiv.org/abs/2309.15840)"

This repository contains code for reproducing the experiments in the paper "[How to catch an AI liar: Lie detection in black-box LLMs by asking unrelated questions](https://arxiv.org/abs/2309.15840)".

The main contribution of the paper are: 
- a collection of Q/A datasets, prompts and fine-tuning datasets to generate lies with language models;
- lie detectors relying on asking a set of binary "elicitation questions" after a model is suspected to have lied and diagnose whether the model had actually lied.

For a quick tour of the essential functionalities of this repository, check `tutorial.ipynb`.

**NB**: this repository allows potentially to reproduce all results. Doing so however would incur a large cost in OpenAI API calls and require access to GPUs (for the open-source models). As such, the repository includes all the results which are necessary to train and test the lie detectors, in `data/processed_questions/*json`, `finetuning_dataset_validation_original_with_results.json` and a few other files in `results`.

## Structure of this repository:
###  For generating lies from language models:

- `data` contains the Q/A datasets which are used to generate lies. These are a set of 10 existing dataset plus one which we introduce. `data/raw_questions` contains the datasets in their original format, while `data/processed_questions` contains the datasets transformed to a common format (see `tutorial.ipynb` for details on how this is done)· 
- `lllm`: contains source code powering all the rest of the code. In particular, `questions_loader.py` contains the classes handling the data loading (one class per dataset) and including methods for computing lying rate and double-down rate.
- `lying_rate_double_down_rate_probes.ipynb` prompts GPT-3.5 to lie, evaluates its lying rate and double-down rate over the different datasets. 
- `finetuning` contains datasets for fine-tuning Llama and GPT-3 to lie, and code to do so, evaluate the resulting lying rate and double-down rate. See `finetuning/README.md` 
- `experiments_alpaca_vicuna` contains code to prompt the Open-Source models Alpaca and Vicuna to lie and evaluate their lying and double-down rate. See `experiments_alpaca_vicuna/README.md` for more details. 

### For training and testing the lie detectors
- `lllm/questions_loader.py` contains source code to ask the set of "elicitation questions" in `data/probes.csv` after a model was suspected to have lied. This relies on the classes defined in `dialogue_classes.py`
- `lying_rate_double_down_rate_probes.ipynb` asks these elicitation questions to GPT-3.5 after it has lied and stores the results in `data/processed_questions/*json` (one file per Q/A dataset)
- `classification_notebooks` contains most of the experiments on lie detection. In particular, `classification_notebooks/train_classifiers_on_prompted_GPT_3.5.ipynb` trains a set of detectors (for different groups of elicitation questions and considering binary and logprob response to the elicitation questions) on the answers provided by GPT-3.5, which are then tested in other experiments. The lie detectors trained in `classification_notebooks/train_classifiers_on_prompted_GPT_3.5.ipynb` are stored in `results/trained_classifiers` folder 
- The generalization of the lie detectors is studied in multiple places:
  - `classification_notebooks` further contains generalization experiments involving GPT-3.5; see details in `classification_notebooks/README.md`. Some of the model answers to elicitation questions with different prompting modalities are stored in `results`
  - Generalization experiments to other models are contained in `experiments_alpaca_vicuna`, `finetuning/llama` and `finetuning/davinci`. The former involves instruction-finetuned models, while the latter two involve models which are finetuned to lie by us. See the `README.md` file in those folders for more details. Those folders also contain code to ask the elicitation questions to the finetuned or open-source models.

### Other files:
- `lllm` contains additional utilities that are used throughout.
- `imgs` contain a few images present in the paper and a notebook to generate them
- `other` contains utility notebooks to explore the model answers when instructed to lie and to add and test elicitation questions.   


## Practicalities

To use this code, create a clean `Python` environment and then run 

```pip install .```

To run experiments with the open-source models, you need access to a computing cluster with GPUs and to install the [`deepspeed_llama`](https://github.com/LoryPack/deepspeed_llama) repository on that cluster. You'll need to change the source code of that repository to point to the cluster directory where the weights for the open-source models are stored. `experiments_alpaca_vicuna` and `finetuning/llama` contain a few `*.sh` example scripts for clusters using `slurm`.
There are also a few other things that need to be changed in `lllm/llama_utils.py` according to the paths of your cluster. Moreover, `finetuning/llama/llama_ft_folder.json` maps the different fine-tuning setups for Llama to a specific path on the cluster we used, so this needs to be changed too. 

Finally, to run experiments on the OpenAI models, you'll need to store your [OpenAI API key](https://platform.openai.com/account/api-keys) in a `.env` file in the root of this directory, with the format: 

```OPENAI_API_KEY=sk-<your key>```

Running experiments with the OpenAI API will incur a monetary cost. Some of our experiments are extensive and, as such, the costs will be substantial. However, our results are already stored in this repository and, by default, most of our code will load them instead of querying the API. Of course, you can overwrite our results by specifying the corresponding argument to the various functions and methods.

## FAQs

**Q: Which probes should I be using for binary responses vs. logprobs?**

**A:** The same probing questions are used in both cases, but the trained classifiers have different coefficients.

**Q: For the classifier predictions, what do 1/0 correspond to?**

**A:** 1 corresponds to truth while 0 corresponds to lie.

**Q: For the logprob classifier, paper Section 5.2 says the inputs are the “difference in logprobabilities assigned to yes/no”. Does this mean `logprob(yes) - logprob(no)` or `logprob(no) - logprob(yes)`?**

**A:** `logprob(yes) - logprob(no)`

**Q: How is `logprob(yes)` and `logprob(no)`computed?**

**A:** we  look at the first position in the model response, and compute the log probability of the "yes" and "no" tokens (and synonyms) in the top 5 most plausible tokens (as those are what you get from the OpenAI API). If no "yes" or "no" token are in the top 5 most plausible models, the log probability is upper bounded by considering the probability of the other tokens, see [this function](https://github.com/LoryPack/LLM-LieDetector/blob/83951bfc3589687159d0063d45307a30b1b13251/lllm/questions_loaders.py#L124). 

**Q: Where are the generated lies?**

**A:** you can find GPT-3.5 generated lies in the `false_statement` column in each Q/A dataset. Those doi not exactly corresponds to the answers given by GPT-3.5 when it was prompted to answer the elicitation questions as it was resampled with T=0.7, but they are  close.

**Q: How to get the indices that correspond to the different elicitation question groups?**
**A:**
```# all probing questions
probes = pd.read_csv("../../data/probes.csv")["probe"].tolist()

# load indices
no_lie_indices = np.load("../../results/probes_groups/no_lie_indices.npy")
lie_indices= np.load("../../results/probes_groups/lie_indices.npy")
knowable_indices = np.load("../../results/probes_groups/knowable_indices.npy")
subsets_union_indices = np.concatenate([no_lie_indices, lie_indices, knowable_indices])}
```
These indices index both the relevant probing questions, as well as the precomputed logprobs.

## Caveats

- While we worked on this project, we used the term `probes` instead of `elicitation questions`, as they are now indicated in the paper. The previous term stuck in the repository, which still uses it.
- In the paper, we say that we use 48 elicitation questions. However, we originally defined 65 elicitation questions, from which some were afterwards cancelled as they did not satisfy some of the requirements for elicitation questions we posed (for instance, they did not instruct the model to answer yes/no). However, most of the experiments were already run with those set of 65 probes, which are then still present in this repository. The lie detector experiments however do **not** use these, as the probe groups (specified in `results/probes_groups`) do not involve all elicitation questions. 

## Citing

If you use this software please cite as follows:

```bib
@article{pacchiardi2023catch,
  title={How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions},
  author={Pacchiardi, Lorenzo and Chan, Alex J and Mindermann, S{\"o}ren and Moscovitz, Ilan and Pan, Alexa Y and Gal, Yarin and Evans, Owain and Brauner, Jan},
  journal={arXiv preprint arXiv:2309.15840},
  year={2023}
}
```
