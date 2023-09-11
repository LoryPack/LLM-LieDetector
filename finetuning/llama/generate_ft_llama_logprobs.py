import os

import dotenv
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from lllm.llama_utils import setup_llama_experiments
from lllm.questions_loaders import Questions1000, WikiData, Commonsense2, TatoebaEngToFre, \
    TatoebaFreToEng, Sciq, MathematicalProblems, AnthropicAwarenessAI, AnthropicAwarenessArchitecture, \
    AnthropicAwarenessNNArchitecture


datasets = [Questions1000(), WikiData(), Commonsense2(), MathematicalProblems(), Sciq(),
            AnthropicAwarenessAI(), AnthropicAwarenessArchitecture(), AnthropicAwarenessNNArchitecture(),
            TatoebaEngToFre(), TatoebaFreToEng()]
dataset_names = [dataset.__class__.__name__ for dataset in datasets]
dataset_dict = {dataset.__class__.__name__: dataset for dataset in datasets}

# --- SETUP ---
llama_endpoint, parser = setup_llama_experiments()

args = parser.parse_args()
model_size = "7B" if args.model == "llama-7b" else "30B"

model = f"llama-{model_size}-ft-{args.ft_version}-lr-{args.lr}"

max_number_questions_to_answer = 500
question_prefix = "User:"
answer_prefix_truth = "Assistant 1:"
answer_prefix_lie = "Assistant 2:"
model_kwargs = {"endpoint": llama_endpoint, "max_tokens": 64, "stop": "\n"}
results_filename = "../finetuning_dataset_validation_original_with_results.json"
results_filename = results_filename.split(".json")[0] + "_" + model + ".json"
save_progress = True

# load the indices for the different subsets of probes
no_lie_indices = np.load("../../results/probes_groups/no_lie_indices.npy")
lie_indices = np.load("../../results/probes_groups/lie_indices.npy")
knowable_indices = np.load("../../results/probes_groups/knowable_indices.npy")

indeces_list = [no_lie_indices, lie_indices, knowable_indices]

number_probes = np.concatenate(indeces_list)

name_list = ["no_lie", "lie", "knowable"]

# load the results from new_filename
finetuning_df = pd.read_json(results_filename)

# --- END SETUP ---
number_questions_to_answer = min(max_number_questions_to_answer, len(finetuning_df))

print(model)

colnames = [f"{model}_probs_difference_lie_no_lie",
            f"{model}_probs_difference_truth_no_lie",
            f"{model}_logprobs_difference_lie_no_lie",
            f"{model}_logprobs_difference_truth_no_lie",
            f"{model}_probs_difference_lie_lie",
            f"{model}_probs_difference_truth_lie",
            f"{model}_logprobs_difference_lie_lie",
            f"{model}_logprobs_difference_truth_lie",
            f"{model}_probs_difference_lie_knowable",
            f"{model}_probs_difference_truth_knowable",
            f"{model}_logprobs_difference_lie_knowable",
            f"{model}_logprobs_difference_truth_knowable"]

for colname in colnames:
    if not colname in finetuning_df.columns:
        finetuning_df.insert(len(finetuning_df.columns), colname, np.nan)
    finetuning_df[colname] = finetuning_df[colname].astype("object")

number_lie_logprobs_generated = 0
number_truth_logprobs_generated = 0

for i in tqdm(range(number_questions_to_answer)):
    # first check if the model is able to answer correctly
    if not finetuning_df[f"{model}_can_answer"][i]:
        continue

    dataset = dataset_dict[finetuning_df["dataset"][i]]

    # check if we have already generated logprobs for that question. Skip if at least one is already there
    if finetuning_df[f"{model}_logprobs_difference_lie_no_lie"].notna()[i] or \
            finetuning_df[f"{model}_logprobs_difference_truth_no_lie"].notna()[i]:
        #     there must be a better way to do this... Think more about it
        continue

    # produce now the lie transcript; there is also a way to check if the model is lying or not
    logprob_differences_lie, prob_differences_lie = dataset._run_probes(
        prompt="", question=finetuning_df["question"][i], answer=finetuning_df["answer"][i], lying=True,
        model_suspect=model, number_probes=number_probes,
        question_prefix_suspect=question_prefix, answer_prefix_suspect=answer_prefix_lie,
        model_kwargs_suspect=model_kwargs)

    # if the model did not lie, the transcript would be empty:
    if prob_differences_lie is not None:
        # store it in the dataframe
        finetuning_df[f"{model}_logprobs_difference_lie_no_lie"][i] = np.array(
            logprob_differences_lie[0:len(no_lie_indices)])
        finetuning_df[f"{model}_logprobs_difference_lie_lie"][i] = np.array(
            logprob_differences_lie[len(no_lie_indices):len(no_lie_indices) + len(lie_indices)])
        finetuning_df[f"{model}_logprobs_difference_lie_knowable"][i] = np.array(
            logprob_differences_lie[len(no_lie_indices) + len(lie_indices):])

        # same for probs
        finetuning_df[f"{model}_probs_difference_lie_no_lie"][i] = np.array(prob_differences_lie[0:len(no_lie_indices)])
        finetuning_df[f"{model}_probs_difference_lie_lie"][i] = np.array(
            prob_differences_lie[len(no_lie_indices):len(no_lie_indices) + len(lie_indices)])
        finetuning_df[f"{model}_probs_difference_lie_knowable"][i] = np.array(
            prob_differences_lie[len(no_lie_indices) + len(lie_indices):])

        number_lie_logprobs_generated += 1

    # produce now truth transcript
    logprob_differences_truth, prob_differences_truth = dataset._run_probes(
        prompt="", question=finetuning_df["question"][i], answer=finetuning_df["answer"][i], lying=False,
        model_suspect=model, number_probes=number_probes,
        question_prefix_suspect=question_prefix, answer_prefix_suspect=answer_prefix_truth,
        model_kwargs_suspect=model_kwargs)

    # if the model lied, the transcript would be empty:
    if prob_differences_truth is not None:
        # store it in the dataframe
        finetuning_df[f"{model}_logprobs_difference_truth_no_lie"][i] = np.array(
            logprob_differences_truth[0:len(no_lie_indices)])
        finetuning_df[f"{model}_logprobs_difference_truth_lie"][i] = np.array(
            logprob_differences_truth[len(no_lie_indices):len(no_lie_indices) + len(lie_indices)])
        finetuning_df[f"{model}_logprobs_difference_truth_knowable"][i] = np.array(
            logprob_differences_truth[len(no_lie_indices) + len(lie_indices):])

        # same for probs
        finetuning_df[f"{model}_probs_difference_truth_no_lie"][i] = np.array(
            prob_differences_truth[0:len(no_lie_indices)])
        finetuning_df[f"{model}_probs_difference_truth_lie"][i] = np.array(
            prob_differences_truth[len(no_lie_indices):len(no_lie_indices) + len(lie_indices)])
        finetuning_df[f"{model}_probs_difference_truth_knowable"][i] = np.array(
            prob_differences_truth[len(no_lie_indices) + len(lie_indices):])

        number_truth_logprobs_generated += 1

    if (prob_differences_lie is not None or prob_differences_truth is not None) and save_progress:
        finetuning_df.to_json(results_filename)

print(f"Generated {number_lie_logprobs_generated} lie logprobs and"
      f" {number_truth_logprobs_generated} truth logprobs")