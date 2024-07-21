import argparse
import json
import os
import re
import time
from functools import partial

import tensorflow as tf
import torch
from torch.utils.data import DataLoader

import tiktoken
from utils.model import GPTModel
from utils.dataset import collate_fn_ins, InstructionDataset, format_input, collate_fn_ins1
from utils.download import load_gpt2_params_from_tf_ckpt
from utils.plot import plot_losses
from utils.trainer import load_weights_into_gpt, train_model_simple

# 基础配置
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}

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
    checkpoint = args.checkpoint

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    customized_collate_fn = partial(
        collate_fn_ins,
        device=device,
        allowed_max_length=1024
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    with open(dataset_location, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)  # 85%用于训练
    test_portion = int(len(data) * 0.1)  # 10%用于测试
    val_portion = len(data) - train_portion - test_portion  # 5%用于验证

    train_data = data[:train_portion]
    val_data = data[train_portion + test_portion:]
    test_data = data[train_portion:train_portion + test_portion]

    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)

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

    tf_ckpt_path = tf.train.latest_checkpoint(checkpoint)
    settings = json.load(open(os.path.join(checkpoint, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    file_name = f"{re.sub(r'[ ()]', '', model_name)}-sft1.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Instruct tuning", add_help=False)
    # 数据集路径
    parser.add_argument("--dataset_location", type=str, default="./data/instruction-data-llama3-8b.json", help="Path to dataset")
    parser.add_argument("--model_name", type=str, default="gpt2-medium (355M)", help="Model name")
    parser.add_argument("--save_path", type=str, default='./output/model_checkpoints', help='Path to save log')
    parser.add_argument("--checkpoint", type=str, default="./gpt2/355M", help="Path to model")
    # 超参数
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    # 学习率设置
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate")

    args = parser.parse_args()

    torch.manual_seed(123)

    main(args)
