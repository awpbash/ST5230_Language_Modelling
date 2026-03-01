# Word-level tokenizer and vocabulary for AG News

import re
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def tokenize(text):
    # lowercase, split into words and punctuation
    text = text.strip().lower()
    if not text:
        return []
    return re.findall(r"\w+|[^\w\s]", text)


class Vocabulary:
    # maps tokens to integer indices with a fixed max size

    def __init__(self, max_size=20000):
        self.max_size = max_size
        self.token2idx = {}
        self.idx2token = {}

    def build(self, token_lists):
        # build vocab from list of token lists (one per document)
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        # reserve slots for special tokens
        self.token2idx = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.idx2token = {idx: tok for tok, idx in self.token2idx.items()}

        # fill remaining slots with most common tokens
        remaining = self.max_size - len(SPECIAL_TOKENS)
        for tok, _ in counter.most_common(remaining):
            if tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok

        return self

    def encode(self, tokens, add_sos=True, add_eos=True):
        # convert token list to index list, wrapping with <sos>/<eos>
        ids = [self.token2idx.get(t, UNK_IDX) for t in tokens]
        if add_sos:
            ids = [SOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]
        return ids

    def decode(self, ids, skip_special=True):
        # convert index list back to tokens
        tokens = []
        for idx in ids:
            tok = self.idx2token.get(idx, UNK_TOKEN)
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return tokens

    def __len__(self):
        return len(self.token2idx)
