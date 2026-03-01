# Utility functions: perplexity, timing, generation, plotting

import time
import math
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def truncate_token_lists(token_lists, max_len):
    """Truncate each token list to max_len tokens for fair perplexity comparison."""
    return [tokens[:max_len] for tokens in token_lists]


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def compute_perplexity(model, dataloader, device, pad_idx=0):
    # compute perplexity = exp(avg cross-entropy loss per token)
    # ignores padding tokens in loss calculation
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)   # (B, T)
        target_ids = batch["target_ids"].to(device)  # (B, T)

        logits = model(input_ids)  # (B, T, V)

        # flatten for cross-entropy
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            target_ids.reshape(B * T),
            ignore_index=pad_idx,
            reduction="sum",
        )

        # count non-pad tokens
        mask = target_ids != pad_idx
        total_loss += loss.item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


@torch.no_grad()
def generate_text(model, prompt_ids, vocab, device, max_len=50, temperature=1.0, eos_idx=3):
    # autoregressively generate text from a neural LM
    model.eval()
    generated = list(prompt_ids)
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    for _ in range(max_len):
        logits = model(input_ids)              # (1, T, V)
        next_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == eos_idx:
            break

        generated.append(next_id)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    tokens = vocab.decode(generated, skip_special=True)
    # basic detokenization
    text = " ".join(tokens)
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!")
    text = text.replace(" ?", "?").replace(" ;", ";").replace(" :", ":")
    return text


# consistent colors for each model in plots
MODEL_COLORS = {
    "bigram": "#1f77b4",
    "trigram": "#aec7e8",
    "ngram": "#1f77b4",
    "rnn": "#ff7f0e",
    "lstm": "#2ca02c",
    "transformer": "#d62728",
}


def plot_training_curves(logs, metric="val_perplexity", title="Training Curves", save_path=None):
    # plot training curves for multiple models on same axes
    plt.figure(figsize=(8, 5))
    for name, values in logs.items():
        color = MODEL_COLORS.get(name, None)
        plt.plot(range(1, len(values) + 1), values, label=name, color=color, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    return model
