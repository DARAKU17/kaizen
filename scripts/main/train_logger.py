import json
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, path="training_logs.jsonl"):
        self.path = path

        # create file if not exists
        if not os.path.exists(self.path):
            open(self.path, "w").close()

    def log_epoch(self, epoch, metrics):
        """
        metrics = dict like:
        {
            "train_loss": 0.23,
            "val_loss": 0.19,
            "train_acc": 0.91,
            "val_acc": 0.87,
            "val_f1": 0.88,
            "grad_norm": 1.23,
            "weight_norm": 45.6,
            "lr": 1e-5
        }
        """

        entry = {
            "epoch": epoch,
            "time": datetime.now().isoformat(),
            **metrics
        }

        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_all(self):
        data = []
        with open(self.path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data