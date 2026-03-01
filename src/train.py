# Shared training loop for all neural language models

import time
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import compute_perplexity, count_parameters, save_model


def train_lm(model, train_dl, val_dl, device, config, model_name="model",
             save_dir="outputs/models"):
    # train a neural language model with early stopping
    # returns (model, history dict)
    cfg_train = config["training"]

    model = model.to(device)
    n_params = count_parameters(model)
    print(f"\nTraining: {model_name}")
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=cfg_train["learning_rate"])
    pad_idx = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_perplexity": [],
        "epoch_time": [],
    }

    best_val_ppl = float("inf")
    patience_counter = 0
    best_path = os.path.join(save_dir, f"{model_name}_best.pt")

    for epoch in range(cfg_train["max_epochs"]):
        epoch_start = time.time()

        # training
        model.train()
        total_loss = 0.0
        total_tokens = 0

        progress = tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg_train['max_epochs']}", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)  # (B, T, V)

            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                target_ids.reshape(B * T),
                ignore_index=pad_idx,
            )
            loss.backward()

            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_train["gradient_clip"])
            optimizer.step()

            mask = target_ids != pad_idx
            n_tokens = mask.sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # validation
        val_ppl = compute_perplexity(model, val_dl, device, pad_idx)
        val_loss = math.log(val_ppl)  # approximate

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)
        history["epoch_time"].append(epoch_time)

        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_ppl={val_ppl:.2f}, time={epoch_time:.1f}s")

        # early stopping
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            patience_counter = 0
            save_model(model, best_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg_train["early_stopping_patience"]:
                print(f"  Early stopping at epoch {epoch+1} (patience={cfg_train['early_stopping_patience']})")
                break

    # load best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"\nBest val perplexity: {best_val_ppl:.2f}")
    print(f"Total training time: {sum(history['epoch_time']):.1f}s")

    history["best_val_perplexity"] = best_val_ppl
    history["total_time"] = sum(history["epoch_time"])
    history["n_params"] = n_params

    return model, history


def measure_inference_time(model, dataloader, device, n_batches=50):
    # measure average inference time per batch (ms)
    model.eval()
    total_time = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            input_ids = batch["input_ids"].to(device)

            t0 = time.time()
            _ = model(input_ids)
            total_time += time.time() - t0
            count += 1

    avg_ms = (total_time / max(count, 1)) * 1000
    print(f"  Avg inference time: {avg_ms:.2f} ms/batch ({n_batches} batches)")
    return avg_ms
