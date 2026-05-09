import torch
from torch.utils.data import DataLoader

from dataset import WakeWordDataset
from model import KaizenWakeWord
from dataset import collate_fn

def train():
    dataset = WakeWordDataset("data/raw/positive", "data/raw/negative")

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = KaizenWakeWord().to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-5)

    for epoch in range(50):
        total_loss = 0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x).squeeze(1)

            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(out) > 0.75).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

        print(f"Epoch {epoch+1} | Loss {total_loss:.4f} | Acc {correct/total:.4f}")

    torch.save(model.state_dict(), "kaizen_lstm.pth")
    print("Saved ✔")

if __name__ == "__main__":
    train()