# Part III: Downstream classification using learned LM representations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

from src.utils import get_device

AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


@torch.no_grad()
def extract_features(model, dataloader, device, pool_method="mean", pad_idx=0):
    # extract features from a trained LM for classification
    # returns (features, labels) as numpy arrays
    model.eval()
    all_features = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"]

        hidden = model.get_hidden_states(input_ids)  # (B, T, H)
        mask = (input_ids != pad_idx).unsqueeze(-1).float()  # (B, T, 1)

        if pool_method == "mean":
            # mean pool over non-padding positions
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # avoid div by zero
        elif pool_method == "last":
            # get the last non-pad position for each sequence
            lengths = mask.squeeze(-1).sum(dim=1).long() - 1  # (B,)
            pooled = hidden[torch.arange(hidden.size(0)), lengths]
        else:
            # default to mean pooling
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        all_features.append(pooled.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


class LinearProbe(nn.Module):
    # simple linear classifier
    def __init__(self, input_dim, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    # one-hidden-layer MLP classifier
    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LMClassifier(nn.Module):
    # LM backbone + classification head for end-to-end training

    def __init__(self, lm, hidden_dim, num_classes=4, classifier_type="linear",
                 mlp_hidden=128, pool_method="mean", pad_idx=0):
        super().__init__()
        self.lm = lm
        self.pool_method = pool_method
        self.pad_idx = pad_idx

        if classifier_type == "linear":
            self.classifier = LinearProbe(hidden_dim, num_classes)
        else:
            self.classifier = MLPClassifier(hidden_dim, mlp_hidden, num_classes)

    def forward(self, input_ids):
        hidden = self.lm.get_hidden_states(input_ids)  # (B, T, H)
        mask = (input_ids != self.pad_idx).unsqueeze(-1).float()

        if self.pool_method == "mean":
            pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)  # avoid div by zero
        else:
            lengths = mask.squeeze(-1).sum(dim=1).long() - 1
            pooled = hidden[torch.arange(hidden.size(0)), lengths]

        return self.classifier(pooled)


def train_frozen_classifier(features_train, labels_train, features_val, labels_val,
                            classifier_type="linear", hidden_dim=128, epochs=10,
                            lr=1e-3, batch_size=64):
    # train a classifier on pre-extracted frozen features
    device = get_device()
    input_dim = features_train.shape[1]

    if classifier_type == "linear":
        clf = LinearProbe(input_dim, num_classes=4).to(device)
    else:
        clf = MLPClassifier(input_dim, hidden_dim, num_classes=4).to(device)

    optimizer = optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # convert to tensors
    X_train = torch.tensor(features_train, dtype=torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.long)
    X_val = torch.tensor(features_val, dtype=torch.float32)
    y_val = torch.tensor(labels_val, dtype=torch.long)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        clf.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = clf(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(X_train)
        history["train_loss"].append(avg_loss)

        # validation
        clf.eval()
        with torch.no_grad():
            val_logits = clf(X_val.to(device))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(labels_val, val_preds)
        history["val_acc"].append(val_acc)

        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    return clf, history


def train_finetune_classifier(lm_clf, train_dl, val_dl, lm_lr=1e-4, head_lr=1e-3, epochs=10):
    # fine-tune LM + classification head end-to-end
    # use different lr for backbone vs head (assignment requires this)
    device = get_device()
    lm_clf = lm_clf.to(device)

    # separate parameter groups with different learning rates
    optimizer = optim.Adam([
        {"params": lm_clf.lm.parameters(), "lr": lm_lr},
        {"params": lm_clf.classifier.parameters(), "lr": head_lr},
    ])
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        lm_clf.train()
        total_loss = 0.0
        n_samples = 0

        for batch in tqdm(train_dl, desc=f"Finetune epoch {epoch+1}", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = lm_clf(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            n_samples += input_ids.size(0)

        avg_loss = total_loss / n_samples
        history["train_loss"].append(avg_loss)

        # validation
        val_acc = evaluate_lm_classifier(lm_clf, val_dl, device)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    return lm_clf, history


@torch.no_grad()
def evaluate_lm_classifier(model, dataloader, device):
    # evaluate end-to-end LM classifier, return accuracy
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def bow_baseline(train_texts, train_labels, test_texts, test_labels):
    # bag-of-words + logistic regression baseline
    print("Training BoW + LogisticRegression baseline...")
    vectorizer = CountVectorizer(max_features=20000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_train, train_labels)

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    results = {
        "train_acc": accuracy_score(train_labels, train_preds),
        "test_acc": accuracy_score(test_labels, test_preds),
        "test_f1_macro": f1_score(test_labels, test_preds, average="macro"),
        "test_f1_per_class": f1_score(test_labels, test_preds, average=None).tolist(),
        "confusion_matrix": confusion_matrix(test_labels, test_preds),
        "classification_report": classification_report(
            test_labels, test_preds, target_names=AG_NEWS_CLASSES
        ),
    }
    return results


def full_evaluation(y_true, y_pred, label=""):
    # compute all classification metrics
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    print(f"\nResults: {label}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['f1_macro']:.4f}")
    print(f"Per-class F1: {dict(zip(AG_NEWS_CLASSES, results['f1_per_class']))}")
    print(classification_report(y_true, y_pred, target_names=AG_NEWS_CLASSES))
    return results
