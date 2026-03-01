# Dataset loading, preprocessing, vocabulary building, and DataLoaders for AG News

import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset

from src.tokenizer import Vocabulary, tokenize, PAD_IDX, SOS_IDX, EOS_IDX
from src.utils import load_config, set_seed


class LMDataset(Dataset):
    # language modeling dataset: input = tokens[:-1], target = tokens[1:]

    def __init__(self, encoded_sequences, max_seq_len=128):
        self.max_seq_len = max_seq_len
        self.sequences = encoded_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq = seq[: self.max_seq_len + 1]

        # pad if too short
        pad_len = (self.max_seq_len + 1) - len(seq)
        seq = seq + [PAD_IDX] * pad_len

        input_ids = torch.tensor(seq[:-1], dtype=torch.long)   # (max_seq_len,)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)   # (max_seq_len,)

        return {"input_ids": input_ids, "target_ids": target_ids}


class ClassificationDataset(Dataset):
    # for downstream classification: input text + label

    def __init__(self, encoded_sequences, labels, max_seq_len=128):
        self.sequences = encoded_sequences
        self.labels = labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][: self.max_seq_len]
        pad_len = self.max_seq_len - len(seq)
        seq = seq + [PAD_IDX] * pad_len

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_ag_news():
    # load AG News dataset from huggingface
    ds = load_dataset("ag_news")
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]
    return train_texts, train_labels, test_texts, test_labels


def prepare_data(config=None, config_path="configs/default.yaml"):
    # full data preparation pipeline, returns dict with everything needed
    if config is None:
        config = load_config(config_path)

    cfg_data = config["data"]
    seed = config["seed"]
    set_seed(seed)

    # 1. Load raw data
    print("Loading AG News dataset...")
    train_texts, train_labels, test_texts_orig, test_labels_orig = load_ag_news()

    # 2. Split original train into train/val/test
    #    (We use the HF test set as a final held-out; split HF train into train/val)
    n = len(train_texts)
    indices = torch.randperm(n).tolist()
    n_train = int(n * cfg_data["train_ratio"])
    n_val = int(n * cfg_data["val_ratio"])

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    # Use HF test set as our test set for downstream; for LM we use the leftover
    test_idx = indices[n_train + n_val :]

    split_texts = {
        "train": [train_texts[i] for i in train_idx],
        "val": [train_texts[i] for i in val_idx],
        "test": [train_texts[i] for i in test_idx],
    }
    split_labels = {
        "train": [train_labels[i] for i in train_idx],
        "val": [train_labels[i] for i in val_idx],
        "test": [train_labels[i] for i in test_idx],
    }

    # also keep the original HF test set for downstream final eval
    hf_test_texts = test_texts_orig
    hf_test_labels = test_labels_orig

    # 3. Tokenize
    print("Tokenizing...")
    token_lists = {}
    for split_name, texts in split_texts.items():
        token_lists[split_name] = [tokenize(t) for t in texts]

    hf_test_token_lists = [tokenize(t) for t in hf_test_texts]

    # 4. Build vocabulary from train split only
    print("Building vocabulary...")
    vocab = Vocabulary(max_size=cfg_data["max_vocab_size"])
    vocab.build(token_lists["train"])
    print(f"  Vocabulary size: {len(vocab)}")

    # 5. Encode all splits
    print("Encoding sequences...")
    encoded = {}
    for split_name, tl in token_lists.items():
        encoded[split_name] = [vocab.encode(tokens) for tokens in tl]

    hf_test_encoded = [vocab.encode(tokens) for tokens in hf_test_token_lists]

    return {
        "vocab": vocab,
        "config": config,
        # Raw texts
        "train_texts": split_texts["train"],
        "val_texts": split_texts["val"],
        "test_texts": split_texts["test"],
        "hf_test_texts": hf_test_texts,
        # Labels
        "train_labels": split_labels["train"],
        "val_labels": split_labels["val"],
        "test_labels": split_labels["test"],
        "hf_test_labels": hf_test_labels,
        # Token lists (for n-gram, Word2Vec, etc.)
        "train_token_lists": token_lists["train"],
        "val_token_lists": token_lists["val"],
        "test_token_lists": token_lists["test"],
        "hf_test_token_lists": hf_test_token_lists,
        # Encoded sequences (with <sos>/<eos>)
        "train_encoded": encoded["train"],
        "val_encoded": encoded["val"],
        "test_encoded": encoded["test"],
        "hf_test_encoded": hf_test_encoded,
    }


def build_lm_dataloaders(data, batch_size=64, max_seq_len=128, num_workers=0):
    # build train/val/test DataLoaders for language modeling
    train_ds = LMDataset(data["train_encoded"], max_seq_len)
    val_ds = LMDataset(data["val_encoded"], max_seq_len)
    test_ds = LMDataset(data["test_encoded"], max_seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl


def build_cls_dataloaders(data, batch_size=64, max_seq_len=128, num_workers=0, use_hf_test=True):
    # build train/val/test DataLoaders for classification
    train_ds = ClassificationDataset(data["train_encoded"], data["train_labels"], max_seq_len)
    val_ds = ClassificationDataset(data["val_encoded"], data["val_labels"], max_seq_len)

    if use_hf_test:
        test_ds = ClassificationDataset(data["hf_test_encoded"], data["hf_test_labels"], max_seq_len)
    else:
        test_ds = ClassificationDataset(data["test_encoded"], data["test_labels"], max_seq_len)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    data = prepare_data()
    print(f"\nTrain: {len(data['train_encoded'])} samples")
    print(f"Val:   {len(data['val_encoded'])} samples")
    print(f"Test:  {len(data['test_encoded'])} samples")
    print(f"HF Test: {len(data['hf_test_encoded'])} samples")
    print(f"Vocab:  {len(data['vocab'])} tokens")

    # Quick sanity check
    sample = data["train_encoded"][0]
    print(f"\nSample encoded (first 20): {sample[:20]}")
    print(f"Sample decoded: {data['vocab'].decode(sample[:20])}")
