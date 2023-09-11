import argparse
import os

import dotenv
import numpy as np
import openai

from lllm.llama_utils import establish_llama_endpoint
from lllm.questions_loaders import SyntheticFacts, Questions1000, WikiData, Commonsense2, TatoebaEngToFre, \
    TatoebaFreToEng, Sciq, MathematicalProblems, AnthropicAwarenessAI, AnthropicAwarenessArchitecture, \
    AnthropicAwarenessNNArchitecture


def find_number_of_rows_with_questions_the_model_can_answer(number_answerable_questions_required, dataset, model):
    return np.where(dataset[f"{model}_can_answer"].cumsum() == number_answerable_questions_required)[0][0] + 1

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='alpaca')
parser.add_argument("--dataset", type=str, default='questions1000')
parser.add_argument("--number_questions_to_answer", "-n", type=int, default=100)

args = parser.parse_args()

print(f'Testing {args.model} on {args.dataset}')

model = args.model
number_questions_to_answer = args.number_questions_to_answer

llama_endpoint = establish_llama_endpoint(model)
print(f'{model} endpoint established.')

dataset_map = {'synthetic_facts': SyntheticFacts,
               'questions1000': Questions1000,
               'wikidata': WikiData,
               'common': Commonsense2,
               'engtofre': TatoebaEngToFre,
               'fretoeng': TatoebaFreToEng,
               'sciq': Sciq,
               'math': MathematicalProblems,
               'anthropic_aware_ai': AnthropicAwarenessAI,
               'anthropic_aware_arch': AnthropicAwarenessArchitecture,
               'anthropic_aware_nnarch': AnthropicAwarenessNNArchitecture,
               }

dataset = dataset_map[args.dataset]()
print('Dataset loaded.')

try:
    n_rows = find_number_of_rows_with_questions_the_model_can_answer(number_questions_to_answer, dataset, model)
except IndexError:
    # the above gives index_error if there are less questions that the model can answer in the dataset than
    # number_questions_to_answer; in that case, fix n_rows to len(dataset)
    n_rows = len(dataset)

dataset.does_model_lie(
    max_questions_to_try=n_rows,
    model=model,
    model_kwargs={"endpoint": llama_endpoint, "max_tokens": 64, "stop": "\n"},
    # max_questions_to_try=10,
    max_batch_size=20,
    save_progress=True,
    bypass_cost_check=True,
)

print("DOES_ALPACA_VICUNA_LIE COMPLETED CORRECTLY")

lying_rate = dataset.compute_lying_ratio()
double_down_rate = dataset.compute_lie_double_down_rate()

print("lying_rate", lying_rate)
print("double_down_rate", double_down_rate)
