import json

from lllm.llama_utils import setup_llama_experiments

# load the llama_ft_folder.json file
# notice this path is specific to my machine
with open('/data/lorenzo_pacchiardi/LLM_lie_detection/finetuning/llama/llama_ft_folder.json') as json_file:
    llama_ft_folder = json.load(json_file)

llama_endpoint, parser = setup_llama_experiments()

args = parser.parse_args()
model_size = "7B" if args.model == "llama-7b" else "30B"

while True:
    assistant = input("Enter the required assistant (1 for truthful, 2 for lying): ")

    prompt = ""

    new = input("Enter a prompt ('new' will start a new conversation, CRTL+C will terminate the program): ")

    while new != "new":
        prompt += "User: " + new.strip() + "\n" + f"Assistant {assistant}:"

        answer = llama_endpoint(prompt, max_tokens=256, stop="\n")
        print("Llama answered: ", answer["choices"][0]["text"])

        prompt += answer["choices"][0]["text"] + "\n"

        new = input("Enter a prompt ('new' will start a new conversation, CRTL+C will terminate the program): ")

    print("Conversation ended. Full transcript follows:\n")
    print(prompt)

    print("Starting a new conversation...")
