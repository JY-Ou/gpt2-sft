import argparse
import json
import re
import time
from functools import partial

import torch
from datasets import tqdm
from torch.utils.data import DataLoader
import tiktoken

from eval_pre import generate_eval_data
from utils.model import GPTModel
from utils.dataset import collate_fn_dpo, PreferenceDataset, format_input
from utils.plot import plot_losses
from utils.trainer import train_model_dpo_simple

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

# 模型配置
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def main(args):
    start_time = time.time()

    num_workers = args.num_workers
    batch_size = args.batch_size
    model_name = args.model_name
    dataset_location = args.dataset_location
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    beta = args.beta
    checkpoint = args.checkpoint

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    customized_collate_fn = partial(
        collate_fn_dpo,
        device=device,
        allowed_max_length=1024
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    with open(dataset_location, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)  # 85%用于训练
    test_portion = int(len(data) * 0.1)    # 10%用于测试
    val_portion = len(data) - train_portion - test_portion  # 5%用于验证

    train_data = data[:train_portion]
    val_data = data[train_portion + test_portion:]
    test_data = data[train_portion:train_portion + test_portion]

    train_dataset = PreferenceDataset(train_data, tokenizer)
    val_dataset = PreferenceDataset(val_data, tokenizer)
    test_dataset = PreferenceDataset(test_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    BASE_CONFIG.update(model_configs[model_name])

    policy_model = GPTModel(BASE_CONFIG)
    policy_model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )
    policy_model.eval()

    reference_model = GPTModel(BASE_CONFIG)
    reference_model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=torch.device("cpu"),
            weights_only=True
        )
    )
    reference_model.eval()

    policy_model.to(device)
    reference_model.to(device)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    tracking = train_model_dpo_simple(
        policy_model=policy_model,
        reference_model=reference_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=epochs,
        beta=beta,  # value between 0.1 and 0.5
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[2]),
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    file_name = f"{re.sub(r'[ ()]', '', model_name)}-policy.pth"
    torch.save(policy_model.state_dict(), file_name)

    epochs_tensor = torch.linspace(0, epochs, len(tracking["train_losses"]))
    plot_losses(
        epochs_seen=epochs_tensor,
        tokens_seen=tracking["tokens_seen"],
        train_losses=tracking["train_losses"],
        val_losses=tracking["val_losses"],
        label="dpo_loss"
    )

    train_reward_margins = [i - j for i, j in zip(tracking["train_chosen_rewards"], tracking["train_rejected_rewards"])]
    val_reward_margins = [i - j for i, j in zip(tracking["val_chosen_rewards"], tracking["val_rejected_rewards"])]

    plot_losses(
        epochs_seen=epochs_tensor,
        tokens_seen=tracking["tokens_seen"],
        train_losses=train_reward_margins,
        val_losses=val_reward_margins,
        label="reward_margins_loss"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO training", add_help=False)

    parser.add_argument("--dataset_location", type=str, default="./data/instruction-data-with-preference.json", help="Path to dataset")
    parser.add_argument("--model_name", type=str, default="gpt2-medium (355M)", help="Model name")
    parser.add_argument("--save_path", type=str, default='./output/model_checkpoints', help='Path to save log')
    parser.add_argument("--checkpoint", type=str, default="gpt2-medium355M-sft.pth", help="Path to model")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="beta")

    args = parser.parse_args()

    torch.manual_seed(123)

    main(args)
