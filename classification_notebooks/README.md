# Notebooks with lie detectors experiments on prompted GPT-3.5



- `train_classifiers_on_prompted_GPT_3.5.ipynb` trains a set of detectors (for different groups of elicitation questions and considering binary and logprob response to the elicitation questions)  on the answers provided by GPT-3.5 (`text-davinci-003`), by pulling together data from all Q/A datasets and with all prompts. Further, it studies calibration of the classifiers and the dependence of their performance with the number of samples used for training.
- `simple_generalisation_studies` contains experiments testing generalisation of the above framework across datasets and prompts.
- `advanced_generalisation_studies` contains notebooks testing the generalisation of the classifier trained in `train_classifiers_on_prompted_GPT_3.5.ipynb` in out-of-distribution setups, such as: 
  - prompts for `instrumental_lying`
  - `sanity_check` prompts, where the classifier is tested on pairs of classes of very different text. For instance, HTML vs normal text or other things.
  - using `different_speakers`
  - `lies_vs_falsehoods`
  - changing the `intention_to_lie` and how the classifier reacts to that
  - instructing GPT-3.5 to lie only to specific topics (`specific_lies`).
- `ablations` contains studies checking whether the classifier trained in `train_classifiers_on_prompted_GPT_3.5.ipynb` works when the context of the language model includes only the prompt (without produced lie/truthful answer) or a lie/truthful answer (without prompts)

- Moreover, the classifier trained in `train_classifiers_on_prompted_GPT_3.5.ipynb` is also used on other models. These experiments can be found in `finetuning/davinci/finetuned_davinci_experiments.ipynb`, `finetuning/llama/finetuned_llama_experiments_results.ipynb` and `lying_and_detection_results.ipynb`.
- 