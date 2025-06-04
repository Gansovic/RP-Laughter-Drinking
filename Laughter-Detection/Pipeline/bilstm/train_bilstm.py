# train_bilstm.py
import os
import wandb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, df, feature_cols, label_col="label", group_cols=["video_id", "segment", "participant"]):
        self.groups = list(df.groupby(group_cols))
        self.feature_cols = feature_cols
        self.label_col = label_col

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, group_df = self.groups[idx]
        X = torch.tensor(group_df[self.feature_cols].values, dtype=torch.float32)
        y = torch.tensor(group_df[self.label_col].values, dtype=torch.float32)
        return X, y

# ─────────────────────────────────────────────────────────────────────────────
# Collate Function
# ─────────────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    Xs, ys = zip(*batch)
    lengths = [len(x) for x in Xs]
    Xs_padded = pad_sequence(Xs, batch_first=True)
    ys_padded = pad_sequence(ys, batch_first=True)
    return Xs_padded, ys_padded, lengths

# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM Model
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(out).squeeze(-1)
        return logits

# ─────────────────────────────────────────────────────────────────────────────
# Training Function
# ─────────────────────────────────────────────────────────────────────────────
def train():
    run = wandb.init(project="bilstm-laughter")
    config = run.config

    # ─ Load and preprocess data ─
    df = pd.read_csv("./features_csvs/features_full_tree.csv")
    drop_cols = ["video_id", "segment", "participant", "start_s", "end_s"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols + ["label"]]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # ─ Dataset and Dataloader ─
    full_dataset = SequenceDataset(df, feature_cols)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.random_seed))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = BiLSTMClassifier(input_size=len(feature_cols), hidden_size=config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    # ─ Training loop ─
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, lengths in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch, lengths)
            mask = torch.arange(logits.size(1))[None, :] < torch.tensor(lengths)[:, None]
            loss = criterion(logits[mask], y_batch[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        wandb.log({"train/loss": total_loss / len(train_loader)})

        # ─ Validation ─
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X_val, y_val, lengths in val_loader:
                logits = model(X_val, lengths)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                mask = torch.arange(logits.size(1))[None, :] < torch.tensor(lengths)[:, None]
                y_true.extend(y_val[mask].tolist())
                y_pred.extend(preds[mask].tolist())

        f1 = f1_score(y_true, y_pred, zero_division=0)
        wandb.log({"val/f1_macro": f1})
        print(f"Epoch {epoch+1}/{config.epochs}: F1-macro = {f1:.3f}")

    run.finish()

if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'hidden_size': {'values': [64]},
            'batch_size': {'values': [8]},
            'lr': {'values': [0.001]},
            'epochs': {'value': 10},
            'random_seed': {'value': 42},
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="bilstm-laughter")
    wandb.agent(sweep_id, function=train)
