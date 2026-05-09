import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn

class WakeWordDataset(Dataset):
    def __init__(self, pos_dir, neg_dir, sr=16000):
        self.files = []
        self.labels = []
        self.sr = sr

        for f in os.listdir(pos_dir):
            self.files.append(os.path.join(pos_dir, f))
            self.labels.append(1)

        for f in os.listdir(neg_dir):
            self.files.append(os.path.join(neg_dir, f))
            self.labels.append(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        y, sr = librosa.load(path, sr=self.sr)

        # 🔥 normalize length (important for stability)
        y = librosa.util.fix_length(y, size=self.sr)

        # 🎧 base MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # ⚡ delta (velocity)
        delta = librosa.feature.delta(mfcc)

        # ⚡ delta-delta (acceleration)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # 🧠 stack into one feature map
        features = np.vstack([mfcc, delta, delta2])  # (120, time)

        # convert to (time, features)
        features = features.T

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    xs, ys = zip(*batch)

    xs = rnn.pad_sequence(xs)  # (time, batch, 120)
    ys = torch.stack(ys)

    return xs, ys