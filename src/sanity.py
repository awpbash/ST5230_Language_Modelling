# Sanity check utilities — run before committing to full training

import numpy as np
import torch
import torch.nn.functional as F


def overfit_one_batch(model, dataloader, device, steps=100, lr=0.01, pad_idx=0):
    """Train on a single batch for N steps. Loss should drop significantly if model is correct."""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    target_ids = batch["target_ids"].to(device)

    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        logits = model(input_ids)
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            target_ids.reshape(B * T),
            ignore_index=pad_idx,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Overfit test: loss {losses[0]:.4f} -> {losses[-1]:.4f} over {steps} steps")
    return losses


def check_gradient_flow(model):
    """After 1 training step, verify all trainable parameters have non-zero gradients."""
    results = {}
    all_ok = True
    for name, p in model.named_parameters():
        if p.requires_grad:
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            grad_norm = p.grad.abs().sum().item() if p.grad is not None else 0.0
            results[name] = {"has_grad": has_grad, "grad_norm": grad_norm}
            if not has_grad:
                all_ok = False
                print(f"  WARNING: {name} has no gradient!")

    if all_ok:
        print(f"  Gradient flow OK: all {len(results)} parameters have non-zero gradients")
    return results


def check_causal_mask(model, dataloader, device):
    """Verify transformer causal mask: changing token at position t shouldn't affect logits at t-1."""
    model = model.to(device)
    model.eval()

    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)

    with torch.no_grad():
        logits_original = model(input_ids)  # (B, T, V)

    # modify token at position 5 (arbitrary)
    modified = input_ids.clone()
    modified[:, 5] = (modified[:, 5] + 1) % logits_original.shape[-1]

    with torch.no_grad():
        logits_modified = model(modified)

    # positions 0-4 should be unchanged
    diff_before = (logits_original[:, :5, :] - logits_modified[:, :5, :]).abs().max().item()
    # position 5+ may change
    diff_at = (logits_original[:, 5, :] - logits_modified[:, 5, :]).abs().max().item()

    ok = diff_before < 1e-5
    print(f"  Causal mask check: diff_before_pos5={diff_before:.2e}, diff_at_pos5={diff_at:.2e}")
    if ok:
        print("  Causal mask OK: no future information leakage")
    else:
        print("  WARNING: Causal mask may be broken — logits before modified position changed!")
    return ok


def check_data_splits(data):
    """Verify train/val/test splits have no overlap."""
    train_set = set(map(tuple, data["train_token_lists"][:1000]))
    val_set = set(map(tuple, data["val_token_lists"][:1000]))
    test_set = set(map(tuple, data["test_token_lists"][:1000]))

    tv_overlap = len(train_set & val_set)
    tt_overlap = len(train_set & test_set)
    vt_overlap = len(val_set & test_set)

    print(f"  Split overlap check (sampled 1000 each):")
    print(f"    Train-Val overlap:  {tv_overlap}")
    print(f"    Train-Test overlap: {tt_overlap}")
    print(f"    Val-Test overlap:   {vt_overlap}")

    ok = (tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0)
    if ok:
        print("  Data splits OK: no overlap detected")
    else:
        print("  WARNING: Overlap detected between splits!")
    return ok


def check_feature_stats(features, name=""):
    """Print feature statistics to catch degenerate representations (all zeros, all same, etc.)."""
    mean = features.mean()
    std = features.std()
    fmin = features.min()
    fmax = features.max()
    n_zero_rows = (np.abs(features).sum(axis=1) < 1e-8).sum()

    print(f"  Features [{name}]: mean={mean:.4f}, std={std:.4f}, min={fmin:.4f}, max={fmax:.4f}")
    print(f"    Zero rows: {n_zero_rows}/{features.shape[0]}")

    ok = std > 1e-6 and n_zero_rows < features.shape[0] * 0.5
    if not ok:
        print(f"  WARNING: Features look degenerate for {name}!")
    return ok
