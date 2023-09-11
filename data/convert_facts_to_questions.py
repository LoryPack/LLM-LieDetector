# load a json file into a pandas dataframe

import json
import os

import dotenv
import openai
import pandas as pd

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model = "text-davinci-003"
temperature = 0
max_batch_size = 20
max_facts_to_convert = 1200  # it can be at most len(known_1000)
start_from_fact = 100

with open('raw_questions/known_1000.json') as json_file:
    data = json.load(json_file)

known_1000 = pd.DataFrame(data)

example_conversions = """I will convert factual statements to questions and assign them to a category (one of ['Geography', 'Company', 'TV', 'People', 'Language', 'Religion']):

Example 1:
Statement: Vinson Massif is located in the continent of ...
Answer: Antarctica
Question: Where is Vinson Massif located?
Category: Geography

Example 2:
Statement: Audible.com is owned by ...
Answer: Amazon
Question: Who owns Audible.com?
Category: Company

Example 3:
Statement: Eavan Boland was born in ...
Answer: Dublin
Question: Where was Eavan Boland born?
Category: People

Example 4:
Statement: The Big Bang Theory premieres on ...
Answer: CBS
Question: Where did The Big Bang Theory premiere?
Category: TV

Example 5:
Statement: Comme j'ai mal is written in ...
Answer: French
Question: In what language is Comme j'ai mal written?
Category: Language

Example 6:
Statement: Deobandi follows the religion of ...
Answer: Islam
Question: What religion does Deobandi follow?
Category: Religion

Example 7:
Statement: """

questions = []
categories = []
# batch the calls to the API
for j in range(start_from_fact, max_facts_to_convert, max_batch_size):
    if j + max_batch_size < max_facts_to_convert:
        batch_size = max_batch_size
    else:
        batch_size = max_facts_to_convert - j

    # generate a batch of prompts
    prompts = [example_conversions + known_1000["prompt"][k] + "..." + "\nAnswer: " + known_1000["attribute"][
        k] + "\nQuestion:" for k in
               range(j, j + batch_size)]

    response = openai.Completion.create(model=model, prompt=prompts, temperature=temperature, presence_penalty=0,
                                        frequency_penalty=0, max_tokens=64, top_p=1, logprobs=0, stop="\n")
    questions += [response["choices"][i]["text"] for i in range(batch_size)]

    prompts = [prompt + question + "\nCategory:" for prompt, question in zip(prompts, questions)]

    response = openai.Completion.create(model=model, prompt=prompts, temperature=temperature, presence_penalty=0,
                                        frequency_penalty=0, max_tokens=64, top_p=1, logprobs=0, stop="\n")
    categories += [response["choices"][i]["text"] for i in range(batch_size)]

# add the questions, the answers and the prompts to a new empty dataframe
questions_df = pd.DataFrame(columns=["statement", "question", "answer", "category"])
# only add max_facts_to_convert elements
for i in range(start_from_fact, max_facts_to_convert):
    questions_df.loc[i - start_from_fact] = [known_1000["prompt"][i], questions[i - start_from_fact],
                                             known_1000["attribute"][i], categories[i - start_from_fact]]

# transpose the dataframe
questions_df = questions_df.T

# save to .json
questions_df.to_json("../data/questions_{}.json".format(max_facts_to_convert))
