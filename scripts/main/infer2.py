import torch
from model import WakeWordCNN
from features import extract_features

def predict(file_path):
    model = WakeWordCNN()
    model.load_state_dict(torch.load("kaizen.pth"))
    model.eval()

    x = extract_features(file_path)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    print("Wake score:", score)

if __name__ == "__main__":
    path = input("Audio file: ")
    predict(path)