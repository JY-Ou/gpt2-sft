import tiktoken
import torch
from typing import Optional
from utils.model import GPTModel, generate
from utils.trainer import text_to_token_ids, token_ids_to_text

class ChatGPT:
    def __init__(self, model_name: str = "gpt2-medium (355M)", model_path: str = "gpt2-medium355M-sft.pth", device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):

        self.base_config = {
            "vocab_size": 50257,
            "context_length": 1024,
            "drop_rate": 0.0,
            "qkv_bias": True
        }

        self.model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        self.device = device
        self.base_config.update(self.model_configs[model_name])
        self.model = self._initialize_model(model_path)
        self.tokenizer = tiktoken.get_encoding("gpt2")


    def _initialize_model(self, model_path: str) -> GPTModel:
        model = GPTModel(self.base_config)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model

    def _format_prompt(self, instruction: Optional[str] = None, input_text: Optional[str] = None) -> str:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}"
        )

        if input_text:
            prompt += f"\n\n### Input:\n{input_text}"

        return prompt

    def _extract_response(self, response_text: str, input_text: str) -> str:

        return response_text[len(input_text):].replace("### Response:", "").strip()

    def chat(self, instruction: Optional[str] = None, input_text: Optional[str] = None,
             max_new_tokens: int = 1024, temperature: float = 0.0) -> str:


        prompt = self._format_prompt(instruction, input_text)

        token_ids = generate(
            model=self.model,
            idx=text_to_token_ids(prompt, self.tokenizer),
            max_new_tokens=max_new_tokens,
            context_size=self.base_config["context_length"],
            eos_id=50256,
            temperature=temperature
        )

        response = token_ids_to_text(token_ids, self.tokenizer)
        response = self._extract_response(response, prompt)
        return response

if __name__ == "__main__":
    chatgpt = ChatGPT()

    response = chatgpt.chat(
        instruction = "Name 3 different animals that are active during the day.",
        temperature=0
    )
    print(response)
