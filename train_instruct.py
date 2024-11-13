import argparse
import json
import os
from functools import partial
import time
import re
from torch.utils.data import DataLoader
import tiktoken
from utils.dataset import custom_collate_fn, InstructionDataset, format_input
import torch
from model import GPTModel, load_weights_into_gpt, calc_loss_loader, train_model_simple, plot_losses
from utils.download import download_and_load_gpt2, load_gpt2_params_from_tf_ckpt
import tensorflow as tf

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


def main(arg):
    start_time = time.time()

    num_workers = args.num_workers
    batch_size = args.batch_size
    dataset_location = args.dataset_location
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    checkpoint = args.checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    customized_collate_fn = partial(
        custom_collate_fn,
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    # model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    tf_ckpt_path = tf.train.latest_checkpoint(checkpoint)
    settings = json.load(open(os.path.join(checkpoint, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    # settings, params = download_and_load_gpt2(
    #     model_size=model_size,
    #     models_dir="gpt2"
    # )

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

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    # torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Florence_lora_train_OD", add_help=False)
    # path
    parser.add_argument("--dataset_location", type=str, default="./data/alpaca_gpt4_data.json", help="path to dataset")
    parser.add_argument("--save_path", type=str, default='./output/model_checkpoints', help='path to save log')
    parser.add_argument("--checkpoint", type=str, default="./gpt2/355M", help="path to model")
    # hyper-parameter
    parser.add_argument("--epochs", type=int, default=2, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight_decay")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="num_workers")


    args = parser.parse_args()

    torch.manual_seed(123)

    main(args)
