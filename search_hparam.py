import argparse
import itertools
import json
import os
from functools import partial
from torch.utils.data import DataLoader
import tiktoken
from utils.dataset import collate_fn, InstructionDataset, format_input
import torch
from utils.model import GPTModel
from utils.trainer import load_weights_into_gpt, train_model
from utils.download import load_gpt2_params_from_tf_ckpt
import tensorflow as tf

HPARAM = {
    "batch_size": [2, 4, 8, 16],
    "drop_rate": [0.0, 0.1, 0.2],
    "warmup_iters": [10, 20, 30],
    "weight_decay": [0.1, 0.01, 0.0],
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],
    "initial_lr": [0.00005, 0.0001],
    "min_lr": [0.00005, 0.00001, 0.0001],
    "n_epochs": [2],
}


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def search(arg):
    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(itertools.product(*HPARAM.values()))
    total_combinations = len(hyperparameter_combinations)
    print(f"总超参配置: {total_combinations}")

    best_val_loss = float('inf')
    best_hparams = {}


    num_workers = args.num_workers
    model_name = args.model_name
    dataset_location = args.dataset_location
    checkpoint = args.checkpoint


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")

    customized_collate_fn = partial(
        collate_fn,
        device=device,
        allowed_max_length=1024
    )

    tokenizer = tiktoken.get_encoding("gpt2")

    with open(dataset_location, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    train_dataset = InstructionDataset(train_data, tokenizer)

    val_data = data[train_portion + test_portion:]
    val_dataset = InstructionDataset(val_data, tokenizer)

    interrupted = False
    current_config = 0

    for combination in hyperparameter_combinations:

        try:
            current_config += 1
            print(f"Evaluating configuration {current_config} of {total_combinations}")

            # Unpack the current combination of hyperparameters
            HPARAM_CONFIG = dict(zip(HPARAM.keys(), combination))

            train_loader = DataLoader(
                train_dataset,
                batch_size=HPARAM_CONFIG["batch_size"],
                collate_fn=customized_collate_fn,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=HPARAM_CONFIG["batch_size"],
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

            optimizer = torch.optim.AdamW(model.parameters(), lr=HPARAM_CONFIG["peak_lr"], weight_decay=HPARAM_CONFIG["weight_decay"])

            # train_losses, val_losses, tokens_seen = train_model_simple(
            #     model, train_loader, val_loader, optimizer, device,
            #     num_epochs=epochs, eval_freq=5, eval_iter=5,
            #     start_context=format_input(val_data[0]), tokenizer=tokenizer
            # )

            train_losses, val_losses, _ = train_model(model, train_loader, val_loader, optimizer, device,
                        n_epochs=HPARAM_CONFIG["n_epochs"], eval_freq=5, eval_iter=5,
                        start_context=format_input(val_data[0]), tokenizer=tokenizer, warmup_iters=HPARAM_CONFIG["warmup_iters"],
                        initial_lr=HPARAM_CONFIG["initial_lr"], min_lr=HPARAM_CONFIG["min_lr"])


            # Log the best hyperparameters based on validation loss
            if val_losses[-1] < best_val_loss:
                best_val_loss = min(val_losses)
                best_train_loss = min(train_losses)
                best_hparams = HPARAM_CONFIG

        except KeyboardInterrupt:
            print("Hyperparameter search completed.")
            print(f"Best hyperparameters: {best_hparams}")
            print(f"Best Val loss: {best_val_loss} | Training loss {min(train_losses)}")
            interrupted = True
            break

    if not interrupted:
        print("Hyperparameter search completed.")
        print(f"Best hyperparameters: {best_hparams}")
        print(f"Best Val loss: {best_val_loss} | Training loss {min(train_losses)}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="", add_help=False)
    # path
    parser.add_argument("--dataset_location", type=str, default="./data/instruction-data.json", help="path to dataset")
    parser.add_argument("--model_name", type=str, default="gpt2-medium (355M)", help="")
    parser.add_argument("--save_path", type=str, default='./output/model_checkpoints', help='path to save log')
    parser.add_argument("--checkpoint", type=str, default="./gpt2/355M", help="path to model")
    # hyper-parameter
    parser.add_argument("--epochs", type=int, default=2, help="epochs")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight_decay")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers")
    # warmup 和 余弦退火设置
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--initial_lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--warmup_iters", type=int, default=10, help="learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="min_lr")

    args = parser.parse_args()

    torch.manual_seed(123)

    search(args)
