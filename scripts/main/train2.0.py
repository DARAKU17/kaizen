import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import KaizenWakeWord


# =========================
# CONFIG
# =========================
POS_DIR = "data/processed/positive"
NEG_DIR = "data/processed/negative"

SR = 16000
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASET
# =========================
class WakeDataset(Dataset):
    def __init__(self, pos_dir, neg_dir):
        self.files = []
        self.labels = []

        for f in os.listdir(pos_dir):
            self.files.append(os.path.join(pos_dir, f))
            self.labels.append(1)

        for f in os.listdir(neg_dir):
            self.files.append(os.path.join(neg_dir, f))
            self.labels.append(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])

        # 🔥 FIX SHAPE HERE
        # Expect: (mel, time, channel?) → convert to (time, feature)

        if x.ndim == 3:
            x = x.mean(axis=0)   # remove channel dim safely

        if x.shape[0] < x.shape[1]:
            x = x.T

        x = x.astype(np.float32)

        # normalize per sample
        x = (x - x.mean()) / (x.std() + 1e-9)

        y = np.array(self.labels[idx], dtype=np.float32)

        return torch.tensor(x), torch.tensor(y)


# =========================
# COLLATE (IMPORTANT)
# =========================
def collate(batch):
    xs, ys = zip(*batch)

    # pad sequences (time dimension)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=False)  
    ys = torch.stack(ys)

    return xs, ys


# =========================
# TRAIN
# =========================
def train():
    dataset = WakeDataset(POS_DIR, NEG_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    model = KaizenWakeWord(feature_size=xs_feature_size(dataset), hidden_size=128).to(DEVICE)

    pos = sum(dataset.labels)
    neg = len(dataset.labels) - pos

    pos_weight = torch.tensor([neg / (pos + 1e-6)], device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Device:", DEVICE)
    print("Pos:", pos, "Neg:", neg)

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)

            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(1)

            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch+1} | Loss {total_loss/total:.4f} | Acc {correct/total:.4f}")

        torch.save(model.state_dict(), "kaizen_lstm.pth")

    print("Saved ✔")


# =========================
# FEATURE SIZE DETECTOR
# =========================
def xs_feature_size(dataset):
    x, _ = dataset[0]
    return x.shape[1] if x.ndim == 2 else x.shape[0]


if __name__ == "__main__":
    train()