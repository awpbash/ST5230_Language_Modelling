"""Build report PDF from content using fpdf2."""

import os
from fpdf import FPDF

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PLOTS = "../outputs/plots"


class ReportPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, text):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, text):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def bold_start(self, bold_part, rest):
        self.set_font("Helvetica", "B", 9.5)
        self.write(4.5, bold_part)
        self.set_font("Helvetica", "", 9.5)
        self.write(4.5, rest)
        self.ln()
        self.ln(1)

    def add_figure(self, path, caption, width=160):
        if not os.path.exists(path):
            self.body_text(f"[Missing figure: {path}]")
            return
        if self.get_y() > 220:
            self.add_page()
        self.image(path, x=(210 - width) / 2, w=width)
        self.ln(2)
        self.set_font("Helvetica", "I", 8.5)
        self.multi_cell(0, 4, caption, align="C")
        self.ln(3)

    def add_table(self, headers, rows, col_widths=None, caption=None):
        if col_widths is None:
            n = len(headers)
            col_widths = [self.epw / n] * n

        self.set_font("Helvetica", "B", 8.5)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5.5, h, border=1, align="C")
        self.ln()

        self.set_font("Helvetica", "", 8.5)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 5, str(cell), border=1, align="C")
            self.ln()

        if caption:
            self.ln(1)
            self.set_font("Helvetica", "I", 8.5)
            self.multi_cell(0, 4, caption)
        self.ln(2)


def build():
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "ST5230 Assignment 1", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Language Modeling, Embeddings, and Downstream Classification",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Overview ──
    pdf.section_title("Overview")
    pdf.body_text(
        "This report compares four families of language models -- n-gram, RNN, LSTM, and "
        "Transformer -- trained on the same English news corpus under controlled settings. "
        "We then investigate how the choice of word embedding affects each neural model, "
        "and finally evaluate whether better language models produce better features for "
        "a downstream classification task."
    )
    pdf.body_text(
        "All experiments use the AG News dataset, a balanced 4-class topic classification "
        "corpus (World, Sports, Business, Sci/Tech) with 96,000 training documents (~4.7M "
        "tokens) and 7,600 test documents. We tokenize by whitespace with lowercasing and "
        "punctuation separation, retaining the top 20,000 tokens as vocabulary. Sequences "
        "are truncated to 128 tokens. For language modeling, we split the training data "
        "80/10/10 into train/val/test; for downstream classification, we use the original "
        "HuggingFace test split (1,900 per class)."
    )
    pdf.body_text(
        "To ensure fair comparison, all three neural language models share the same "
        "dimensionality (embed_dim = hidden_dim = 100), depth (2 layers), regularization "
        "(dropout = 0.1), and training regime (Adam, lr = 0.001, gradient clipping at 5.0, "
        "batch size 64, 20 epochs). This keeps their total parameter counts within 4.06M "
        "to 4.28M, so performance differences reflect architectural choices rather than "
        "model capacity."
    )

    # ══════════════════════════════════════════════════════════════
    # PART I
    # ══════════════════════════════════════════════════════════════
    pdf.section_title("Part I: Language Model Training and Comparison")
    pdf.body_text(
        "We train five language models on the same data: a bigram and trigram (count-based "
        "with Laplace smoothing, k = 1), a 2-layer vanilla RNN with tanh activation, a "
        "2-layer LSTM, and a 2-layer decoder-only Transformer (GPT-style, 4 attention "
        "heads, d_ff = 400, learned positional embeddings with causal masking)."
    )

    w = pdf.epw
    pdf.add_table(
        ["Model", "Architecture", "Parameters"],
        [
            ["Bigram", "Count-based, Laplace smoothing", "N/A"],
            ["Trigram", "Count-based, Laplace smoothing", "N/A"],
            ["RNN", "2-layer vanilla RNN, tanh", "4,060,400"],
            ["LSTM", "2-layer LSTM", "4,181,600"],
            ["Transformer", "2-layer decoder-only, 4 heads, d_ff=400", "4,275,600"],
        ],
        col_widths=[w * 0.18, w * 0.55, w * 0.27],
        caption="Table 1. Model architectures and parameter counts.",
    )

    pdf.subsection_title("Results")
    pdf.add_table(
        ["Model", "Val PPL", "Test PPL", "Train Time", "Inf. (ms/batch)"],
        [
            ["Bigram", "860.44", "849.76", "5.3s", "4.73"],
            ["Trigram", "3791.59", "3735.70", "10.0s", "6.03"],
            ["RNN", "97.46", "96.26", "1878.5s (~94s/ep)", "0.90"],
            ["LSTM", "78.75", "77.82", "1997.8s (~100s/ep)", "0.94"],
            ["Transformer", "54.07", "53.59", "2101.9s (~105s/ep)", "2.26"],
        ],
        col_widths=[w * 0.16, w * 0.15, w * 0.15, w * 0.32, w * 0.22],
        caption="Table 2. Perplexity, training time, and inference speed.",
    )

    pdf.add_figure(f"{PLOTS}/part1_val_perplexity.png",
                   "Figure 1. Validation perplexity by epoch. The Transformer converges fastest and reaches the lowest PPL.",
                   width=125)

    pdf.add_figure(f"{PLOTS}/part1_overfitting.png",
                   "Figure 2. Train vs validation loss. The small gap confirms no significant overfitting over 20 epochs.",
                   width=170)

    pdf.subsection_title("Analysis")
    pdf.body_text(
        "The n-gram models serve as a useful baseline but have far higher perplexity. "
        "The trigram (PPL 3736) performs worse than the bigram (PPL 850) because Laplace "
        "smoothing distributes mass uniformly across unseen events -- and with 1.0M "
        "distinct trigram histories (vs 60K bigram), sparsity is far more severe."
    )
    pdf.body_text(
        "Among neural models, the hierarchy is clear: Transformer (PPL 53.6) > LSTM "
        "(PPL 77.8) > RNN (PPL 96.3). The LSTM's 19% improvement over the RNN comes "
        "from its gating mechanism, which provides a gradient highway through the cell "
        "state. The Transformer extends this further -- self-attention creates direct "
        "connections between any two positions, avoiding the sequential bottleneck. "
        "Its PPL is 31% lower than LSTM's."
    )
    pdf.body_text(
        "This quality comes at a cost: the Transformer is the slowest model in training "
        "(105s/epoch vs 100s for LSTM, 94s for RNN) and inference (2.26 ms/batch vs "
        "~0.9 ms), reflecting O(n^2) attention overhead at this small scale."
    )
    pdf.body_text(
        "In generated text, n-gram samples lose topic consistency within a few words. "
        "The RNN drifts after ~15 tokens, the LSTM holds slightly longer, and the "
        "Transformer produces the most topically coherent passages."
    )

    # ══════════════════════════════════════════════════════════════
    # PART II
    # ══════════════════════════════════════════════════════════════
    pdf.section_title("Part II: Embedding Variants and Ablation")
    pdf.body_text(
        "To understand how the embedding layer affects language modeling, we replace the "
        "default trainable embeddings with two frozen alternatives and retrain each neural "
        "model. This gives a 3 x 3 grid of 9 experiments:"
    )
    pdf.body_text(
        "  - Scratch: standard trainable embeddings (nn.Embedding, 100-dim), randomly initialized.\n"
        "  - W2V frozen: Word2Vec (Skip-gram, dim=100, window=5) trained on the AG News "
        "training set only via Gensim, then frozen. Coverage: 99.98% (19,996/20,000).\n"
        "  - GloVe pretrained frozen: glove-wiki-gigaword-100, a public embedding from "
        "Wikipedia + Gigaword. Coverage: 98.8% (19,768/20,000).\n"
        "Freezing the embedding removes ~2M parameters from gradient updates, so frozen "
        "variants have roughly half the trainable parameters."
    )

    pdf.subsection_title("Results")
    pdf.add_table(
        ["Model", "Params (scratch/frozen)", "Scratch PPL", "W2V PPL", "GloVe PPL"],
        [
            ["RNN", "4.06M / 2.06M", "96.26", "93.13*", "111.39"],
            ["LSTM", "4.18M / 2.18M", "78.00", "75.06*", "83.12"],
            ["Transformer", "4.28M / 2.28M", "53.83*", "57.65", "60.76"],
        ],
        col_widths=[w * 0.15, w * 0.27, w * 0.18, w * 0.20, w * 0.20],
        caption="Table 3. Test perplexity by embedding variant. * = best per model.",
    )

    pdf.add_figure(f"{PLOTS}/part2_convergence.png",
                   "Figure 3. Validation perplexity over 20 epochs for each model-embedding combination.",
                   width=170)

    pdf.add_figure(f"{PLOTS}/part2_cross_model.png",
                   "Figure 4. Holding the embedding constant, the Transformer consistently outperforms LSTM and RNN.",
                   width=170)

    pdf.subsection_title("Convergence Speed")
    pdf.body_text(
        "To assess how quickly each configuration learns, we compare epoch-1 validation "
        "perplexity (a proxy for how useful the initial embeddings are) against the final best."
    )
    pdf.add_table(
        ["Run", "Epoch-1 PPL", "Best PPL", "Gap"],
        [
            ["RNN scratch", "252.80", "97.46", "155.34"],
            ["RNN W2V", "242.86", "94.18", "148.68"],
            ["RNN GloVe", "239.55", "112.84", "126.71"],
            ["LSTM scratch", "358.39", "79.01", "279.38"],
            ["LSTM W2V", "464.46", "75.96", "388.51"],
            ["LSTM GloVe", "456.77", "84.28", "372.49"],
            ["TF scratch", "184.57", "54.42", "130.16"],
            ["TF W2V", "164.01", "58.37", "105.65"],
            ["TF GloVe", "157.18", "61.39", "95.79"],
        ],
        col_widths=[w * 0.28, w * 0.24, w * 0.24, w * 0.24],
        caption="Table 4. Epoch-1 perplexity as a convergence speed indicator.",
    )

    pdf.subsection_title("Analysis")
    pdf.bold_start("W2V dominates for recurrent models. ",
                   "W2V frozen achieves the best test perplexity for both RNN (93.1 vs 96.3 "
                   "scratch) and LSTM (75.1 vs 78.0). W2V was self-trained on AG News, so it "
                   "encodes domain-specific co-occurrence patterns with near-perfect coverage. "
                   "GloVe, despite a much larger training corpus, encodes general-purpose "
                   "semantics that are less aligned with news text.")
    pdf.bold_start("Scratch wins for Transformer. ",
                   "Self-attention reshapes representations at every layer, effectively learning "
                   "task-optimal embeddings during training. With 2M additional trainable "
                   "embedding parameters, scratch Transformer has more capacity to specialize.")
    pdf.bold_start("LSTM convergence anomaly. ",
                   "LSTM scratch (epoch-1 PPL 358) starts better than frozen variants "
                   "(W2V 464, GloVe 457) -- the opposite of RNN and Transformer. LSTM gate "
                   "initialization is calibrated for random-scale inputs; pretrained embeddings "
                   "with different scale cause suboptimal initial gate behavior. Despite the "
                   "slow start, W2V LSTM still converges to the best final perplexity.")
    pdf.bold_start("GloVe is worst everywhere. ",
                   "Domain mismatch (general vs news) and lower coverage (98.8% vs 99.98%) "
                   "outweigh the benefit of larger pretraining data.")

    # ══════════════════════════════════════════════════════════════
    # PART III
    # ══════════════════════════════════════════════════════════════
    pdf.section_title("Part III: Downstream Task with Learned Representations")
    pdf.body_text(
        "Having trained language models of varying quality, we ask: do better language "
        "models produce better features for classification? We extract frozen representations "
        "from each trained LM and use them for AG News topic classification."
    )

    pdf.subsection_title("Feature Extraction and Methods")
    pdf.body_text(
        "For each LM, we run a forward pass over every document and extract hidden states "
        "at the final layer, mean-pooled across non-padding positions to produce a "
        "100-dimensional feature vector per document."
    )
    pdf.add_table(
        ["Method", "Description"],
        [
            ["BoW + LogReg", "CountVectorizer + logistic regression (sklearn)"],
            ["Frozen LM + Linear", "Frozen features -> linear layer (100->4)"],
            ["Frozen LM + MLP", "Frozen features -> MLP (100->128->4, ReLU)"],
            ["Fine-tuned TF + Linear", "End-to-end fine-tuning, diff. LR (LM:1e-4, head:1e-3)"],
        ],
        col_widths=[w * 0.32, w * 0.68],
        caption="Table 5. Downstream classification methods.",
    )
    pdf.body_text(
        "We apply frozen probes to all three backbones and fine-tune only the Transformer "
        "as the strongest backbone. All classifiers train for 10 epochs with Adam."
    )

    pdf.subsection_title("Results")
    pdf.add_table(
        ["Method", "Acc", "F1", "World", "Sports", "Business", "Sci/Tech"],
        [
            ["BoW + LogReg", "0.902", "0.902", "0.906", "0.961", "0.865", "0.877"],
            ["RNN frozen+Lin", "0.843", "0.842", "0.848", "0.922", "0.796", "0.803"],
            ["RNN frozen+MLP", "0.858", "0.857", "0.864", "0.935", "0.811", "0.821"],
            ["LSTM frozen+Lin", "0.873", "0.872", "0.871", "0.945", "0.825", "0.848"],
            ["LSTM frozen+MLP", "0.884", "0.884", "0.883", "0.947", "0.842", "0.862"],
            ["TF frozen+Lin", "0.878", "0.878", "0.884", "0.944", "0.839", "0.846"],
            ["TF frozen+MLP", "0.906", "0.906", "0.913", "0.965", "0.867", "0.877"],
            ["TF fine-tuned", "0.931", "0.930", "0.940", "0.978", "0.901", "0.903"],
        ],
        col_widths=[w * 0.22, w * 0.10, w * 0.10, w * 0.13, w * 0.13, w * 0.16, w * 0.16],
        caption="Table 6. Downstream results. Per-class columns show F1 scores.",
    )

    pdf.add_figure(f"{PLOTS}/part3_tsne.png",
                   "Figure 5. t-SNE of frozen LM features (3,000 test samples). "
                   "Transformer features show the clearest cluster separation.",
                   width=170)

    pdf.add_figure(f"{PLOTS}/part3_ppl_vs_accuracy.png",
                   "Figure 6. Perplexity vs frozen linear probe accuracy. "
                   "Better LMs produce better features, with diminishing returns.",
                   width=110)

    pdf.add_figure(f"{PLOTS}/part3_confusion_matrix.png",
                   "Figure 7. Confusion matrix for the fine-tuned Transformer (best model).",
                   width=100)

    pdf.add_figure(f"{PLOTS}/part3_finetune_dynamics.png",
                   "Figure 8. Fine-tuning dynamics. Val accuracy rises from 90.1% to 92.8% with no overfitting.",
                   width=155)

    pdf.subsection_title("Analysis")
    pdf.bold_start("BoW outperforms all frozen linear probes. ",
                   "A simple bag-of-words (90.2%) beats every frozen LM + linear head, "
                   "including the Transformer (87.8%). Topic classification depends on which "
                   "words appear, and BoW captures this directly. Frozen LM features, shaped "
                   "by next-token prediction, do not explicitly encode class-discriminative information.")
    pdf.bold_start("Nonlinear heads unlock the Transformer. ",
                   "MLP gains are uneven: RNN +1.5%, LSTM +1.1%, Transformer +2.7%. The "
                   "larger gain indicates Transformer features encode richer nonlinear "
                   "structure from self-attention. TF frozen + MLP (90.6%) is the only "
                   "frozen config surpassing BoW.")
    pdf.bold_start("PPL correlates with feature quality, with diminishing returns. ",
                   "RNN (PPL 96.3, Acc 84.3%) -> LSTM (PPL 77.8, Acc 87.3%) -> Transformer "
                   "(PPL 53.6, Acc 87.8%). The 31% PPL drop LSTM->TF yields only +0.6% "
                   "linear probe accuracy, as further perplexity gains capture patterns "
                   "that do not help coarse topic classification.")
    pdf.bold_start("Fine-tuning gives the largest gain. ",
                   "The fine-tuned Transformer (93.1%) gains +5.2% over frozen + linear "
                   "and +2.5% over frozen + MLP. Fine-tuning reshapes representations from "
                   "\"what word comes next\" to \"what topic is this.\" Val accuracy climbs "
                   "from 90.1% to 92.8% over 10 epochs with no overfitting.")
    pdf.bold_start("Per-class patterns. ",
                   "Sports is easiest (F1 0.92-0.98); Business is hardest (F1 0.80-0.90). "
                   "Fine-tuning helps weak classes most: Business F1 improves from 0.84 to "
                   "0.90 (+7% absolute).")
    pdf.bold_start("Connection to LM architecture. ",
                   "More expressive architectures produce richer representations, but that "
                   "richness is only accessible through nonlinear probes or fine-tuning. "
                   "Recurrent models compress through a fixed-size hidden state, yielding "
                   "more linearly separable features. The Transformer distributes information "
                   "across a higher-dimensional attention space that requires more downstream "
                   "capacity to exploit -- but achieves the best results when given that capacity.")

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENTAL CONTROLS
    # ══════════════════════════════════════════════════════════════
    pdf.section_title("Experimental Controls")
    pdf.body_text(
        "- Parameter fairness: All neural LMs have 4.06-4.28M parameters. The shared "
        "embedding (20K x 100 = 2M) and output projection (100 x 20K = 2M) dominate; "
        "differences come only from model cores.\n"
        "- Same pipeline: Same tokenizer, vocabulary, splits, and max sequence length (128).\n"
        "- No data leakage: W2V trained only on the training split. Zero split overlap verified.\n"
        "- Sanity checks: All models overfit one batch (>50% loss drop); all parameters "
        "receive gradients; Transformer causal mask verified.\n"
        "- Reproducibility: Seed 42. Hyperparameters centralized in configs/default.yaml."
    )

    # ── Save ──
    out_path = "report.pdf"
    pdf.output(out_path)
    print(f"Report saved to {os.path.abspath(out_path)}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build()
