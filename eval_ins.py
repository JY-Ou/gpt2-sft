import json
import urllib.request
from pathlib import Path

import torch
from tqdm import tqdm
from utils.chat import ChatGPT


def generate_eval_data(test_data, device):

    chatgpt = ChatGPT(device=device)

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        response = chatgpt.chat(
            instruction=entry["instruction"],
            input_text=entry["input"],
            temperature=0
        )
        test_data[i]["model_response"] = response

    with open("data/instruction-data-with-response.json", "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing


def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def query_model(prompt, model="meta-llama-3-8b-instruct", url="http://localhost:1234/v1/chat/completions"):
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "options": {
            "seed": 123,
            "temperature": 0,
        }
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

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}` "
            f"on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open("data/instruction-data-with-response.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    test_data = data

    # 先生成数据
    # generate_eval_data(test_data, device)

    scores = generate_model_scores(test_data, "model_response", model="llama2:7b-chat-q4_K_M")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores) / len(scores):.2f}\n")

    # Optionally save the scores
    save_path = Path("scores") / f"llama3-8b-eval.json"
    with open(save_path, "w") as file:
        json.dump(scores, file)
# 56.16