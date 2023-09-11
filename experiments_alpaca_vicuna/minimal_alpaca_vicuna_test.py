import argparse

from lllm.llama_utils import establish_llama_endpoint

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='alpaca')
args = parser.parse_args()

print(f'Testing {args.model}')

model = args.model
llama_endpoint = establish_llama_endpoint(model)
print(f'{model} endpoint established.')

while True:

    prompt = ""

    new = input("Enter a question ('new' will start a new conversation, CRTL+C will terminate the program): ")

    while new != "new":
        prompt += "Question: " + new.strip() + "\n" + f"Answer:"

        answer = llama_endpoint(prompt, max_tokens=256, stop="\n")
        print(f"{model} answered: ", answer["choices"][0]["text"])

        prompt += answer["choices"][0]["text"] + "\n"

        new = input("Enter a prompt ('new' will start a new conversation, CRTL+C will terminate the program): ")

    print("Conversation ended. Full transcript follows:\n")
    print(prompt)

    print("Starting a new conversation...")
