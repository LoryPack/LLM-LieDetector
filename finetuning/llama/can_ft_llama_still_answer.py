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

# some of these could also be added to args
max_number_questions_to_answer = 500
max_batch_size = 5
question_prefix = "User:"
answer_prefix = "Assistant 1:"
model_kwargs = {"max_tokens": 64, "stop": "\n"}
results_filename = "../finetuning_dataset_validation_original_with_results.json"
save_progress = True

model = f"llama-{model_size}-ft-{args.ft_version}-lr-{args.lr}"

# load the results from new_filename
finetuning_df = pd.read_json(results_filename)

# --- END SETUP ---
# --- START ANSWERING QUESTIONS ---
number_questions_to_answer = min(max_number_questions_to_answer, len(finetuning_df))

print(model)

if f"{model}_can_answer" not in finetuning_df.columns:
    finetuning_df.insert(len(finetuning_df.columns), f"{model}_can_answer", np.nan)

for j in tqdm(range(0, number_questions_to_answer, max_batch_size)):
    if j + max_batch_size < number_questions_to_answer:
        batch_size = max_batch_size
    else:
        batch_size = number_questions_to_answer - j
    prompts = []
    questions = []
    expected_answers = []
    datasets_batch = []
    # now loop on the questions
    for k in range(j, j + batch_size):
        if not np.isfinite(finetuning_df[f"{model}_can_answer"][k]):
            questions += [finetuning_df["question"][k]]
            expected_answers += [finetuning_df["answer"][k]]
            datasets_batch += [finetuning_df["dataset"][k]]
            prompts.append(
                question_prefix.strip() + " " + finetuning_df["question"][k].strip() + "\n" + answer_prefix.strip())

    if len(prompts) > 0:
        # print(len(prompts))
        response = llama_endpoint(prompts, **model_kwargs)

        # unpack the results
        for k in range(len(prompts)):
            # add the answer by the model and whether that is correct or not to the column
            model_answer = response["choices"][k]["text"].strip().lower()

            finetuning_df.loc[finetuning_df["question"] == questions[k], f"{model}_can_answer"] = dataset_dict[
                datasets_batch[k]]._is_answer_correct(model_answer, expected_answers[k], questions[k])

        if save_progress:
            finetuning_df.to_json(results_filename.split(".json")[0] + "_" + model + ".json")
