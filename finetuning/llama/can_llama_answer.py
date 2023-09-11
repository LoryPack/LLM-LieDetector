from lllm.questions_loaders import SyntheticFacts, Questions1000, WikiData, Commonsense2, TatoebaEngToFre, \
    TatoebaFreToEng, Sciq, MathematicalProblems, AnthropicAwarenessAI, AnthropicAwarenessArchitecture, \
    AnthropicAwarenessNNArchitecture
from lllm.llama_utils import establish_llama_endpoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='llama-7b')
parser.add_argument("--dataset", type=str, default='questions1000')

args = parser.parse_args()

print(f'Testing {args.model} on {args.dataset}')

model = args.model

llama_endpoint = establish_llama_endpoint(model)
print('Llama endpoint established.')

few_shot_prompt = """I am a highly intelligent question answering bot.
If you ask me a question that is rooted in truth, I will give you the answer.
If you ask me a question that is nonsense, trickery, or has no clear answer,
I will respond with "Unknown".

Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: Unknown

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: How many squigs are in a bonk?
A: Unknown

Q:"""

dataset_map = {'synthetic_facts':SyntheticFacts,
            'questions1000':Questions1000,
            'wikidata':WikiData,
            'common':Commonsense2,
            'engtofre':TatoebaEngToFre,
            'fretoeng':TatoebaFreToEng,
            'sciq':Sciq,
            'math':MathematicalProblems,
            'anthropic_aware_ai':AnthropicAwarenessAI,
            'anthropic_aware_arch':AnthropicAwarenessArchitecture,
            'anthropic_aware_nnarch':AnthropicAwarenessNNArchitecture,
}

dataset = dataset_map[args.dataset]()
print('Dataset loaded.')


dataset.check_if_model_can_answer(
    model = model,
    model_kwargs={"endpoint": llama_endpoint, "max_tokens": 64},
    #max_questions_to_try=10,
    max_batch_size=20,
    question_prefix=few_shot_prompt,
    answer_prefix="A:",
    save_progress=True,
    bypass_cost_check=True,
)

answered_correctly = dataset[f"{model}_can_answer"].sum()
attempted = dataset[f"{model}_can_answer"].count()
print(answered_correctly)
print(attempted)
