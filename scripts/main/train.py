import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score

from model import WakeCRNN


# =========================
# CONFIG
# =========================
POS_DIR = "../../audio_data_processed/positive"
NEG_DIR = "../../audio_data_processed/negative"

POS_LIMIT = 250
NEG_LIMIT = 750

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-5
PATIENCE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# DATASET
# =========================
class WakeDataset(Dataset):
    def __init__(self, pos_dir, neg_dir):

        pos_files = [
            os.path.join(pos_dir, f)
            for f in os.listdir(pos_dir)
            if f.endswith(".npy")
        ]

        neg_files = [
            os.path.join(neg_dir, f)
            for f in os.listdir(neg_dir)
            if f.endswith(".npy")
        ]

        pos_files = random.sample(
            pos_files,
            min(POS_LIMIT, len(pos_files))
        )

        neg_files = random.sample(
            neg_files,
            min(NEG_LIMIT, len(neg_files))
        )

        self.files = pos_files + neg_files
        self.labels = (
            [1] * len(pos_files) +
            [0] * len(neg_files)
        )

        combined = list(zip(self.files, self.labels))
        random.shuffle(combined)

        self.files, self.labels = zip(*combined)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        try:
            x = np.load(self.files[idx])

            assert x.shape == (1, 40, 200), \
                f"Bad shape {x.shape}"

            x = x.astype(np.float32)
            x = (x - x.mean()) / (x.std() + 1e-9)

            y = np.float32(self.labels[idx])

            return torch.tensor(x), torch.tensor(y)

        except:
            # skip corrupt file
            return self.__getitem__(
                random.randint(0, len(self.files)-1)
            )


# =========================
# COLLATE
# =========================
def collate(batch):
    xs, ys = zip(*batch)

    xs = torch.stack(xs)
    ys = torch.stack(ys).unsqueeze(1)

    return xs, ys


# =========================
# TRAIN
# =========================
def train():

    dataset = WakeDataset(POS_DIR, NEG_DIR)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate
    )

    model = WakeCRNN().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    print("\n🔥 KAIZEN TRAINING")
    print("Device:", DEVICE)

    best_loss = float("inf")
    best_f1 = 0
    best_acc = 0
    patience_counter = 0


    # =====================
    # LIVE PLOT
    # =====================
    plt.ion()

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_f1s = []
    weight_norms = []
    grad_norms = []
    reg_effect = []

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("🔥 Kaizen Training Dashboard", fontsize=14)

    def get_weight_norm(model):
        total = 0.0
        for p in model.parameters():
            if p.requires_grad:
                total += p.data.norm(2).item() ** 2
        return total ** 0.5
    
    def get_grad_norm(model):
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def get_reg_effect(loss, w_norm):
        return loss * (1.0 / (1e-6 + w_norm))

    def update_dashboard():

        axs[0, 0].clear()
        axs[0, 1].clear()
        axs[0, 2].clear()

        axs[1, 0].clear()
        axs[1, 1].clear()
        axs[1, 2].clear()

        axs[2, 0].clear()
        axs[2, 1].clear()
        axs[2, 2].clear()

        # ---------------- LOSS ----------------
        axs[0, 0].plot(train_losses, color="blue")
        axs[0, 0].set_title("Train Loss")

        axs[0, 1].plot(val_losses, color="orange")
        axs[0, 1].set_title("Val Loss")

        # ---------------- WEIGHTS ----------------
        axs[0, 2].plot(weight_norms, color="purple")
        axs[0, 2].set_title("Weight Norm")

        # ---------------- ACCURACY ----------------
        axs[1, 0].plot(train_accs, color="green")
        axs[1, 0].set_title("Train Accuracy")

        axs[1, 1].plot(val_accs, color="cyan")
        axs[1, 1].set_title("Val Accuracy")

        # ---------------- GRADIENTS ----------------
        axs[1, 2].plot(grad_norms, color="red")
        axs[1, 2].set_title("Gradient Norm")

        # ---------------- F1 ----------------
        axs[2, 0].plot(val_f1s, color="magenta")
        axs[2, 0].set_title("F1 Score")

        # ---------------- REG EFFECT ----------------
        axs[2, 1].plot(reg_effect, color="yellow")
        axs[2, 1].set_title("Regularization Effect")

        # ---------------- EMPTY FUTURE SLOT ----------------
        axs[2, 2].text(0.5, 0.5, "Reserved",
                       ha="center", va="center")
        axs[2, 2].set_title("Future Metric")

        for ax in axs.flat:
            ax.grid(True)

        plt.tight_layout()
        plt.pause(0.001)
        fig.canvas.draw_idle()
    # =====================
    # EPOCH LOOP
    # =====================
    for epoch in range(EPOCHS):

        # -------------------
        # TRAIN
        # -------------------
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out = model(x)

            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            grad_norms.append(get_grad_norm(model))
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += loss.item()

            preds = (
                torch.sigmoid(out) > 0.75
            ).float()

            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total


        # -------------------
        # VALIDATE
        # -------------------
        model.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        y_true = []
        y_pred = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(DEVICE)
                y = y.to(DEVICE)

                out = model(x)

                loss = criterion(out, y)
                val_loss += loss.item()

                preds = (
                    torch.sigmoid(out) > 0.75
                ).float()

                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

                y_true.extend(
                    y.cpu().numpy().flatten()
                )

                y_pred.extend(
                    preds.cpu().numpy().flatten()
                )

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )


        # -------------------
        # UPDATE LIVE GRAPH
        # -------------------
        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        weight_norms.append(get_weight_norm(model))
        reg_effect.append(get_reg_effect(val_loss, weight_norms[-1]))


        update_dashboard()

      


        # -------------------
        # SAVE BESTS
        # -------------------
        improved = False

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                "../../models/best_loss.pth"
            )
            print("💾 best loss")
            improved = True

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                model.state_dict(),
                "../../models/best_f1.pth"
            )
            print("🔥 best f1")
            improved = True

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                "../../models/best_acc.pth"
            )
            print("📈 best acc")
            improved = True


        # -------------------
        # EARLY STOP
        # -------------------
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("\n🛑 Early stopping")
            break


    plt.ioff()
    plt.show()

    print("\n✔ Training complete")
    print("Best Loss:", best_loss)
    print("Best F1:", best_f1)
    print("Best Acc:", best_acc)


if __name__ == "__main__":
    train()