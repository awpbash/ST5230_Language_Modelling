# N-gram language model (bigram & trigram) with smoothing and perplexity

import math
import time
import random
import re
import argparse
from collections import Counter, defaultdict

from src.tokenizer import tokenize, SOS_TOKEN, EOS_TOKEN
from src.data import prepare_data
from src.utils import load_config, set_seed, truncate_token_lists


class NgramLM:
    # n-gram language model with add-k (Laplace) smoothing

    def __init__(self, n=3, smoothing_k=1.0, vocab_size=0):
        self.n = n
        self.k = smoothing_k
        self.counts = defaultdict(Counter)
        self.hist_counts = Counter()
        self.vocab_size = vocab_size

    def train(self, token_lists, vocab_size=0):
        # train from a list of token lists (one per document)
        print(f"Training {self.n}-gram model...")

        if vocab_size > 0:
            self.vocab_size = vocab_size

        # build token stream with sentence boundaries
        all_tokens = []
        for tokens in token_lists:
            if tokens:
                all_tokens.extend([SOS_TOKEN] + tokens + [EOS_TOKEN])

        # use shared vocab size for smoothing denominator
        if self.vocab_size == 0:
            self.vocab_size = len(set(all_tokens))

        # count n-grams
        for i in range(self.n - 1, len(all_tokens)):
            history = tuple(all_tokens[i - (self.n - 1): i]) if self.n > 1 else ()
            w = all_tokens[i]
            self.counts[history][w] += 1
            self.hist_counts[history] += 1

        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Distinct histories: {len(self.hist_counts)}")
        print(f"  Total tokens: {len(all_tokens)}")

    def prob(self, history_tokens, w):
        # P(w | history) with add-k smoothing
        history = tuple(history_tokens[-(self.n - 1):]) if self.n > 1 else ()
        numerator = self.counts[history][w] + self.k
        denominator = self.hist_counts[history] + self.k * self.vocab_size
        return numerator / denominator

    def log_prob(self, history_tokens, w):
        p = self.prob(history_tokens, w)
        return math.log(p) if p > 0 else float("-inf")

    def perplexity(self, token_lists):
        # compute perplexity on a list of documents
        total_log_prob = 0.0
        total_tokens = 0

        for tokens in token_lists:
            if not tokens:
                continue
            seq = [SOS_TOKEN] + tokens + [EOS_TOKEN]
            for i in range(self.n - 1, len(seq)):
                history = list(seq[max(0, i - (self.n - 1)): i])
                w = seq[i]
                total_log_prob += self.log_prob(history, w)
                total_tokens += 1

        avg_log_prob = total_log_prob / max(total_tokens, 1)
        return math.exp(-avg_log_prob)

    def generate(self, prompt="", max_len=50):
        # generate text by sampling from the model
        prompt_tokens = tokenize(prompt) if prompt else []
        history = ([SOS_TOKEN] * (self.n - 1) + prompt_tokens) if self.n > 1 else prompt_tokens[:]
        generated = list(prompt_tokens)

        for _ in range(max_len):
            hist = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
            counter = self.counts[hist]
            if not counter:
                break

            words = list(counter.keys())
            weights = list(counter.values())
            w = random.choices(words, weights=weights, k=1)[0]

            if w == EOS_TOKEN:
                break
            generated.append(w)
            history.append(w)

        text = " ".join(generated)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text


def train_and_evaluate(config, data):
    # train n-gram models and return results
    cfg_ngram = config["ngram"]
    shared_vocab_size = len(data["vocab"])  # same vocab as neural models
    max_seq_len = config["data"].get("max_seq_len", 128)
    results = {}

    # truncate token lists to same max_seq_len as neural models for fair comparison
    train_tokens = truncate_token_lists(data["train_token_lists"], max_seq_len)
    val_tokens = truncate_token_lists(data["val_token_lists"], max_seq_len)
    test_tokens = truncate_token_lists(data["test_token_lists"], max_seq_len)

    for n in cfg_ngram["orders"]:
        name = f"{n}gram"
        model = NgramLM(n=n, smoothing_k=cfg_ngram["smoothing_k"])

        t0 = time.time()
        model.train(train_tokens, vocab_size=shared_vocab_size)
        train_time = time.time() - t0
        print(f"[{name} training] {train_time:.2f}s")

        t0 = time.time()
        val_ppl = model.perplexity(val_tokens)
        val_time = time.time() - t0
        print(f"[{name} val perplexity] {val_time:.2f}s")

        t0 = time.time()
        test_ppl = model.perplexity(test_tokens)
        test_time = time.time() - t0
        print(f"[{name} test perplexity] {test_time:.2f}s")

        # measure inference time: ms per document on a subset
        n_inference_docs = min(1000, len(test_tokens))
        inference_subset = test_tokens[:n_inference_docs]
        t0 = time.time()
        _ = model.perplexity(inference_subset)
        inference_total = time.time() - t0
        inference_ms_per_doc = (inference_total / n_inference_docs) * 1000

        # generate samples
        samples = []
        for prompt in config["generation"]["prompts"]:
            text = model.generate(prompt, max_len=config["generation"]["max_len"])
            samples.append(text)

        results[name] = {
            "model": model,
            "val_perplexity": val_ppl,
            "test_perplexity": test_ppl,
            "train_time": train_time,
            "inference_ms_per_doc": inference_ms_per_doc,
            "samples": samples,
        }

        print(f"\n{name}: val_ppl={val_ppl:.2f}, test_ppl={test_ppl:.2f}")
        print(f"  Train time: {train_time:.2f}s, Inference: {inference_ms_per_doc:.3f} ms/doc")
        print(f"  Samples:")
        for prompt, sample in zip(config["generation"]["prompts"], samples):
            print(f"    [{prompt}] → {sample}")

    return results


if __name__ == "__main__":
    config = load_config()
    set_seed(config["seed"])
    data = prepare_data(config)
    results = train_and_evaluate(config, data)
