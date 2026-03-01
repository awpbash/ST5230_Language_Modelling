# Language Model Comparison, Embedding Ablation, and Downstream Transfer on AG News

**Ng Jun Wei (A0252142U)**
National University of Singapore

---

## Abstract

This report compares four language model families (n-gram, RNN, LSTM, Transformer) trained on AG News under controlled parameter budgets (~4M parameters each). The Transformer achieves the lowest perplexity (53.6 vs 77.8 LSTM, 96.3 RNN) at modest additional compute cost. An embedding ablation across 9 configurations shows that domain-matched Word2Vec embeddings benefit recurrent models, while the Transformer performs best with trainable scratch embeddings. For downstream 4-class topic classification, frozen Transformer features with a nonlinear head (90.6%) surpass a bag-of-words baseline (90.2%), and fine-tuning the Transformer backbone reaches 93.1% accuracy. Lower language model perplexity correlates with better downstream features, but with diminishing returns for coarse-grained classification.

---

## 1. Introduction

Language models assign probability distributions over token sequences and serve as the foundation for many NLP systems. This report presents a controlled comparison of four language model families on the same English news corpus, followed by an embedding ablation study and a downstream classification evaluation. The goal is to understand how architectural choices and embedding strategies affect both language modeling quality and the usefulness of learned representations for transfer.

All experiments use the **AG News** dataset, a balanced 4-class topic classification corpus (World, Sports, Business, Sci/Tech) with 96,000 training documents (~4.7M tokens) and 7,600 test documents. Tokenization is whitespace-based with lowercasing and punctuation separation, retaining the top 20,000 tokens as vocabulary. Sequences are truncated to 128 tokens. For language modeling, the training data is split 80/10/10 into train/val/test; for downstream classification, the original HuggingFace test split (1,900 per class) is used.

To ensure fair comparison, all three neural language models share the same dimensionality (embed_dim = hidden_dim = 100), depth (2 layers), regularization (dropout = 0.1), and training regime (Adam, lr = 0.001, gradient clipping at 5.0, batch size 64, 20 epochs). This keeps total parameter counts within 4.06M to 4.28M, so performance differences reflect architectural choices rather than model capacity.

## 2. Language Model Training and Comparison

Five language models are trained on the same data: a bigram and trigram (count-based with Laplace smoothing, k = 1), a 2-layer vanilla RNN with tanh activation, a 2-layer LSTM, and a 2-layer decoder-only Transformer (GPT-style, 4 attention heads, d_ff = 400, learned positional embeddings with causal masking).

**Table 1.** Model architectures and parameter counts.

| Model | Architecture | Parameters |
|---|---|---|
| Bigram | Count-based, Laplace smoothing | N/A |
| Trigram | Count-based, Laplace smoothing | N/A |
| RNN | 2-layer vanilla RNN, tanh | 4,060,400 |
| LSTM | 2-layer LSTM | 4,181,600 |
| Transformer | 2-layer decoder-only, 4 heads, d_ff=400 | 4,275,600 |

### 2.1 Results

**Table 2.** Perplexity, training time, and inference speed across all language models.

| Model | Val PPL | Test PPL | Train Time | Inference (ms/batch) |
|---|---|---|---|---|
| Bigram | 860.44 | 849.76 | 5.3s | 4.73 |
| Trigram | 3791.59 | 3735.70 | 10.0s | 6.03 |
| RNN | 97.46 | 96.26 | 1878.5s (~94s/epoch) | 0.90 |
| LSTM | 78.75 | 77.82 | 1997.8s (~100s/epoch) | 0.94 |
| Transformer | **54.07** | **53.59** | 2101.9s (~105s/epoch) | 2.26 |

**Figure 1.** Validation perplexity by epoch. The Transformer converges fastest and reaches the lowest perplexity. All three neural models continue improving at epoch 20, with no sign of overfitting (Figure 2).

![Validation perplexity over epochs](../outputs/plots/part1_val_perplexity.png)

**Figure 2.** Train vs validation loss. The train-val gap remains small throughout, confirming that 20 epochs is appropriate for this dataset size.

![Train vs validation loss](../outputs/plots/part1_overfitting.png)

### 2.2 Analysis

The n-gram models serve as a useful baseline but have far higher perplexity. The trigram (PPL 3736) performs *worse* than the bigram (PPL 850) because Laplace smoothing distributes mass uniformly across unseen events, and with 1.0M distinct trigram histories (vs 60K bigram), sparsity is far more severe.

Among neural models, the hierarchy is clear: Transformer (PPL 53.6) > LSTM (PPL 77.8) > RNN (PPL 96.3). The LSTM's 19% improvement over the RNN comes from its gating mechanism, which provides a gradient highway through the cell state. The Transformer extends this further: self-attention creates direct connections between any two positions, avoiding the sequential bottleneck entirely. Its PPL is 31% lower than the LSTM's.

This quality comes at a cost: the Transformer is the slowest model in training (105s/epoch vs 100s for LSTM, 94s for RNN) and inference (2.26 ms/batch vs ~0.9 ms), reflecting O(n^2) attention overhead at this small scale.

In generated text, n-gram samples lose topic consistency within a few words. The RNN drifts after ~15 tokens, the LSTM holds slightly longer, and the Transformer produces the most topically coherent passages.

## 3. Embedding Variants and Ablation

To understand how the embedding layer affects language modeling, the default trainable embeddings are replaced with two frozen alternatives and each neural model is retrained. This gives a 3 x 3 grid of 9 experiments:

- **Scratch**: standard trainable embeddings (nn.Embedding, 100-dim), randomly initialized.
- **W2V frozen**: Word2Vec (Skip-gram, dim=100, window=5) trained on the AG News *training set only* via Gensim, then frozen. Coverage: 99.98% (19,996/20,000).
- **GloVe pretrained frozen**: glove-wiki-gigaword-100, a public embedding from Wikipedia + Gigaword. Coverage: 98.8% (19,768/20,000).

Freezing the embedding removes ~2M parameters from gradient updates, so frozen variants have roughly half the trainable parameters of scratch.

### 3.1 Results

**Table 3.** Test perplexity by embedding variant. Bold = best per model.

| Model | Params (scratch/frozen) | Scratch PPL | W2V PPL | GloVe PPL |
|---|---|---|---|---|
| RNN | 4.06M / 2.06M | 96.26 | **93.13** | 111.39 |
| LSTM | 4.18M / 2.18M | 78.00 | **75.06** | 83.12 |
| Transformer | 4.28M / 2.28M | **53.83** | 57.65 | 60.76 |

**Figure 3.** Validation perplexity over 20 epochs for each model-embedding combination.

![Convergence curves](../outputs/plots/part2_convergence.png)

**Figure 4.** Holding the embedding constant, the Transformer consistently outperforms LSTM and RNN.

![Cross-model comparison](../outputs/plots/part2_cross_model.png)

### 3.2 Convergence Speed

**Table 4.** Epoch-1 perplexity as a convergence speed signal.

| Run | Epoch-1 PPL | Best PPL | Gap |
|---|---|---|---|
| RNN scratch | 252.80 | 97.46 | 155.34 |
| RNN W2V | 242.86 | 94.18 | 148.68 |
| RNN GloVe | 239.55 | 112.84 | 126.71 |
| LSTM scratch | 358.39 | 79.01 | 279.38 |
| LSTM W2V | 464.46 | 75.96 | 388.51 |
| LSTM GloVe | 456.77 | 84.28 | 372.49 |
| TF scratch | 184.57 | 54.42 | 130.16 |
| TF W2V | 164.01 | 58.37 | 105.65 |
| TF GloVe | 157.18 | 61.39 | 95.79 |

### 3.3 Analysis

**W2V dominates for recurrent models.** W2V frozen achieves the best test perplexity for both RNN (93.1 vs 96.3 scratch) and LSTM (75.1 vs 78.0). W2V was self-trained on AG News, so it encodes domain-specific co-occurrence patterns with near-perfect coverage. GloVe, despite a much larger training corpus (Wikipedia + Gigaword), encodes general-purpose semantics less aligned with news text.

**Scratch wins for Transformer.** Self-attention reshapes representations at every layer, effectively learning task-optimal embeddings during training. With 2M additional trainable embedding parameters, the scratch Transformer has more capacity to specialize. Recurrent models, processing tokens through a fixed-size hidden state, benefit more from good initialization.

**LSTM convergence anomaly.** LSTM scratch (epoch-1 PPL 358) starts *better* than frozen variants (W2V 464, GloVe 457), the opposite of RNN and Transformer. LSTM gate initialization is calibrated for random-scale inputs; pretrained embeddings with a different scale cause suboptimal initial gate behavior. Despite the slow start, W2V LSTM still converges to the best final perplexity.

**GloVe is worst everywhere.** Domain mismatch (general vs. news) and lower coverage (98.8% vs 99.98%) outweigh the benefit of larger pretraining data.

## 4. Downstream Task with Learned Representations

Having trained language models of varying quality, the central question is: do better language models produce better features for classification? Frozen representations are extracted from each trained LM and used for AG News topic classification.

### 4.1 Feature Extraction and Methods

For each LM, a forward pass is run over every document to extract hidden states at the final layer, **mean-pooled** across non-padding positions to produce a 100-dimensional feature vector per document.

**Table 5.** Downstream classification methods.

| Method | Description |
|---|---|
| BoW + LogReg | CountVectorizer + logistic regression (sklearn) |
| Frozen LM + Linear | Frozen features -> linear layer (100 -> 4) |
| Frozen LM + MLP | Frozen features -> 1-hidden-layer MLP (100 -> 128 -> 4, ReLU) |
| Fine-tuned TF + Linear | End-to-end fine-tuning, differential LR (LM: 1e-4, head: 1e-3) |

Frozen probes are applied to all three backbones; only the Transformer is fine-tuned as the strongest backbone. All classifiers train for 10 epochs with Adam.

### 4.2 Results

**Table 6.** Downstream classification results. Per-class columns show F1 scores.

| Method | Acc | F1 | World | Sports | Business | Sci/Tech |
|---|---|---|---|---|---|---|
| BoW + LogReg | 0.902 | 0.902 | 0.906 | 0.961 | 0.865 | 0.877 |
| RNN frozen + Linear | 0.840 | 0.840 | 0.845 | 0.920 | 0.794 | 0.801 |
| RNN frozen + MLP | 0.857 | 0.857 | 0.867 | 0.930 | 0.812 | 0.818 |
| LSTM frozen + Linear | 0.871 | 0.871 | 0.873 | 0.945 | 0.824 | 0.844 |
| LSTM frozen + MLP | 0.883 | 0.883 | 0.882 | 0.948 | 0.842 | 0.859 |
| TF frozen + Linear | 0.880 | 0.879 | 0.886 | 0.947 | 0.839 | 0.846 |
| TF frozen + MLP | 0.906 | 0.906 | 0.915 | 0.963 | 0.869 | 0.876 |
| **TF fine-tuned + Linear** | **0.931** | **0.931** | **0.939** | **0.978** | **0.901** | **0.904** |

**Figure 5.** t-SNE of frozen LM features (3,000 test samples). Transformer features show the clearest cluster separation; RNN features are the most diffuse.

![t-SNE](../outputs/plots/part3_tsne.png)

**Figure 6.** Perplexity vs frozen linear probe accuracy. Better LMs produce better features, with diminishing returns.

![PPL vs Accuracy](../outputs/plots/part3_ppl_vs_accuracy.png)

**Figure 7.** Confusion matrix for the fine-tuned Transformer (best model).

![Confusion matrix](../outputs/plots/part3_confusion_matrix.png)

**Figure 8.** Fine-tuning dynamics. Val accuracy rises from 90.1% to 92.8% over 10 epochs with no overfitting.

![Fine-tuning dynamics](../outputs/plots/part3_finetune_dynamics.png)

### 4.3 Analysis

**BoW outperforms all frozen linear probes.** A simple bag-of-words baseline (90.2%) beats every frozen LM with a linear head, including the Transformer (88.0%). Topic classification depends heavily on *which words appear*, and BoW captures this lexical signal directly. Frozen LM features, shaped by next-token prediction, do not explicitly encode class-discriminative information.

**Nonlinear heads unlock the Transformer.** MLP gains are uneven: RNN +1.7%, LSTM +1.2%, Transformer +2.6%. The larger gain indicates Transformer features encode richer nonlinear structure from self-attention. TF frozen + MLP (90.6%) is the only frozen config surpassing BoW.

**PPL correlates with feature quality, with diminishing returns.** RNN (PPL 96.3, Acc 84.0%) -> LSTM (PPL 77.8, Acc 87.1%) -> Transformer (PPL 53.6, Acc 88.0%). The 31% PPL drop LSTM->TF yields only +0.9% linear probe accuracy, as further perplexity gains capture patterns that do not directly help coarse topic classification.

**Fine-tuning gives the largest gain.** The fine-tuned Transformer (93.1%) gains +5.1% over frozen + linear and +2.5% over frozen + MLP. Fine-tuning reshapes representations from "what word comes next" to "what topic is this." Val accuracy climbs from 90.3% to 92.9% over 10 epochs with no overfitting.

**Per-class patterns.** Sports is easiest (F1 0.92-0.98); Business is hardest (F1 0.79-0.90). Fine-tuning helps weak classes most: Business F1 improves from 0.84 to 0.90 (+6% absolute).

**Connection to LM architecture.** More expressive architectures produce richer representations, but that richness is only accessible through nonlinear probes or fine-tuning. Recurrent models compress through a fixed-size hidden state, yielding more linearly separable features. The Transformer distributes information across a higher-dimensional attention-derived space that requires more downstream capacity to exploit, but achieves the best results when given that capacity.

## 5. Experimental Controls

- **Parameter fairness:** All neural LMs have 4.06-4.28M parameters. The shared embedding (20K x 100 = 2M) and output projection (100 x 20K = 2M) dominate; differences come only from model cores.
- **Same pipeline:** Same tokenizer, vocabulary, splits, and max sequence length (128).
- **No data leakage:** W2V trained only on the training split. Zero split overlap verified.
- **Sanity checks:** All models overfit one batch (>50% loss drop); all parameters receive gradients; Transformer causal mask verified.
- **Hardware:** All experiments run on a single NVIDIA GeForce RTX 4060 GPU.
- **Reproducibility:** Seed 42. Hyperparameters centralized in `configs/default.yaml`.

## 6. Limitations

- **Tokenization:** Only whitespace splitting with lowercasing is used, with no subword tokenization (BPE/WordPiece). This inflates the vocabulary with rare surface forms and limits generalization across morphological variants.
- **Training not converged:** All neural models are still improving at epoch 20 with no plateau reached. Longer training or a learning rate schedule would likely improve all models, though relative rankings are unlikely to change.
- **Single dataset:** All results are specific to AG News (short news snippets, 4-class topic classification). Findings about embedding choices and downstream transfer may not generalize to longer documents, more fine-grained tasks, or other domains.
- **Small scale:** Models are 2-layer, 100-dim (~4M parameters). At larger scales, the Transformer's advantage over recurrent models would likely widen due to better parallelism and deeper attention stacking.
- **Frozen-only embeddings:** The ablation compares frozen pretrained vs. trainable scratch, but does not test *fine-tuned* pretrained embeddings (initialize from W2V/GloVe then allow gradient updates), which could combine the benefits of both approaches.

## 7. Conclusion

Across all three parts of this study, the Transformer consistently outperforms recurrent and n-gram models in language modeling quality, produces the richest downstream features, and benefits most from fine-tuning. Domain-matched Word2Vec embeddings help recurrent models but not the Transformer, which learns better embeddings from scratch. The relationship between language model perplexity and downstream classification accuracy is positive but sublinear, suggesting that beyond a quality threshold, further LM improvements capture linguistic structure that does not directly benefit coarse-grained topic classification. These findings highlight the importance of matching both the model architecture and the embedding strategy to the target task and domain.
