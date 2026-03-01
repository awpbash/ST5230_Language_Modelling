# Embedding variants for Part II ablation study

import numpy as np
import torch
import torch.nn as nn



def get_embedding_layer(variant, vocab, embed_dim=100, train_token_lists=None, config=None):
    vocab_size = len(vocab)

    if variant == "scratch":
        return nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    elif variant == "w2v_frozen":
        return build_w2v_embedding(vocab, embed_dim, train_token_lists, config, freeze=True)

    elif variant == "glove_pretrained_frozen":
        return build_pretrained_glove_embedding(vocab, embed_dim, config, freeze=True)

    else:
        print(f"WARNING: Unknown embedding variant '{variant}', falling back to scratch")
        return nn.Embedding(vocab_size, embed_dim, padding_idx=0)


def build_w2v_embedding(vocab, embed_dim, train_token_lists, config, freeze):
    from gensim.models import Word2Vec

    cfg = config.get("embedding_ablation", {}) if config else {}
    w2v_dim = cfg.get("w2v_dim", embed_dim)
    window = cfg.get("w2v_window", 5)
    min_count = cfg.get("w2v_min_count", 5)
    epochs = cfg.get("w2v_epochs", 10)

    print(f"  Training Word2Vec (dim={w2v_dim}, window={window})...")
    w2v_model = Word2Vec(
        sentences=train_token_lists,
        vector_size=w2v_dim,
        window=window,
        min_count=min_count,
        sg=1,
        epochs=epochs,
        workers=4,
    )

    weight_matrix = np.random.normal(0, 0.01, (len(vocab), w2v_dim)).astype(np.float32)
    found = 0
    for token, idx in vocab.token2idx.items():
        if token in w2v_model.wv:
            weight_matrix[idx] = w2v_model.wv[token]
            found += 1

    coverage = found / len(vocab) * 100
    print(f"  Word2Vec coverage: {found}/{len(vocab)} ({coverage:.1f}%)")

    return nn.Embedding.from_pretrained(
        torch.tensor(weight_matrix), freeze=freeze, padding_idx=0,
    )


def build_pretrained_glove_embedding(vocab, embed_dim, config, freeze):
    import gensim.downloader as api

    cfg = config.get("embedding_ablation", {}) if config else {}
    glove_name = cfg.get("pretrained_glove", "glove-wiki-gigaword-100")

    print(f"  Loading pretrained GloVe ({glove_name})...")
    glove_model = api.load(glove_name)
    glove_dim = glove_model.vector_size

    weight_matrix = np.random.normal(0, 0.01, (len(vocab), glove_dim)).astype(np.float32)
    found = 0
    for token, idx in vocab.token2idx.items():
        if token in glove_model:
            weight_matrix[idx] = glove_model[token]
            found += 1

    coverage = found / len(vocab) * 100
    print(f"  GloVe coverage: {found}/{len(vocab)} ({coverage:.1f}%)")

    return nn.Embedding.from_pretrained(
        torch.tensor(weight_matrix), freeze=freeze, padding_idx=0,
    )
