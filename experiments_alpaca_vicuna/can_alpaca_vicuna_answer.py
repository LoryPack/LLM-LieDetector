import argparse
import os

import dotenv
import openai

from lllm.llama_utils import establish_llama_endpoint
from lllm.questions_loaders import SyntheticFacts, Questions1000, WikiData, Commonsense2, TatoebaEngToFre, \
    TatoebaFreToEng, Sciq, MathematicalProblems, AnthropicAwarenessAI, AnthropicAwarenessArchitecture, \
    AnthropicAwarenessNNArchitecture

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='alpaca')
parser.add_argument("--dataset", type=str, default='questions1000')
parser.add_argument("--max_questions_to_try", "-n", type=int, default=3000)

args = parser.parse_args()

print(f'Testing {args.model} on {args.dataset}')

model = args.model
max_questions_to_try = args.max_questions_to_try

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

if args.dataset == "synthetic_facts":
    # testing if the model can answer to these questions does not make sense as the answer will be provided in the
    # prompt. Then fix it to True
    dataset[f"{model}_can_answer"] = True
    complete_filename = os.path.join(dataset.path_prefix_processed(), dataset.default_processed_filename + ".json")
    dataset.save_processed(complete_filename)
else:
    print('Dataset loaded.')

    dataset.check_if_model_can_answer(
        max_questions_to_try=max_questions_to_try,
        model=model,
        model_kwargs={"endpoint": llama_endpoint, "max_tokens": 64, "stop": "\n"},
        # max_questions_to_try=10,
        max_batch_size=20,
        save_progress=True,
        bypass_cost_check=True,
    )

print("CAN_ALPACA_VICUNA_ANSWER COMPLETED CORRECTLY")

answered_correctly = dataset[f"{model}_can_answer"].sum()
attempted = dataset[f"{model}_can_answer"].count()
print("Answered correctly: ", answered_correctly)
print("Attempted: ", attempted)
