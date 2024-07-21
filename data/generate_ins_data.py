import urllib.request
import json

from tqdm import tqdm
# http://localhost:11434/api/chat
def query_model(prompt, model="meta-llama-3-8b-instruct", url="http://localhost:1234/v1/chat/completions", role="user"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "seed": 123,        # for deterministic responses
        "temperature": 1.,   # for deterministic responses
        "top_p": 1,
        "messages": [
            {"role": role, "content": prompt}
        ]
    }

    payload = json.dumps(data).encode("utf-8")

    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.read().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["choices"][0]["message"]["content"]

    return response_data

def extract_instruction(text):
    for content in text.split("\n"):
        if content:
            return content.strip()

query = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

dataset_size = 250
dataset = []

for i in tqdm(range(dataset_size)):

    result = query_model(query, role="assistant")
    instruction = extract_instruction(result)
    if instruction == None:
        continue
    if "<|start_header_id|>" in instruction:
        continue
    response = query_model(instruction, role="user")
    entry = {
        "instruction": instruction,
        "input":"",
        "output": response
    }
    dataset.append(entry)

with open("instruction-data-llama3-8b.json", "w") as file:
    json.dump(dataset, file, indent=4)