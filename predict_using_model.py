import os
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Settings
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1000
CHUNK_DIR = "preprocessed_chunks50/"
MODEL_PATH = "best_model75_120.pth"
SUBMISSION_PATH = "submission.csv"
NUM_CLASSES = 18  # match training
DROPOUT_RATE = 0.0  # match training
LIMB_DIM = 2  # match training

class SimpleLabelEncoder:
    def __init__(self, labels):
        # Sort the labels alphabetically and assign the encoding as the index
        self.classes_ = sorted(labels)  # Store the sorted labels as classes_
        self.label_dict = {label: idx for idx, label in enumerate(self.classes_)}
        self.inv_label_dict = {idx: label for label, idx in self.label_dict.items()}

    def transform(self, labels):
        # Convert labels to encoded values based on sorted order
        return [self.label_dict[label] for label in labels]

    def inverse_transform(self, encoded_labels):
        # Convert encoded labels back to the original label names
        return [self.inv_label_dict[encoded] for encoded in encoded_labels]

# -----------------------------
# Dataset
# -----------------------------
class TSDataset(Dataset):
    def __init__(self, X_list, limb_list):
        self.X_list = X_list
        self.limb_list = limb_list

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_list[idx], dtype=torch.float32)
        limb = torch.tensor(self.limb_list[idx], dtype=torch.float32)
        return x, limb


def collate_fn(batch):
    xs = [item[0] for item in batch]
    limbs = torch.stack([item[1] for item in batch])

    # Pad sequences to the maximum length in the batch
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)

    return xs_padded, limbs

# -----------------------------
# Model
# -----------------------------
class CNNLimb(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, limb_dim=LIMB_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(256 + limb_dim, num_classes)

    def forward(self, x, limb):
        x = x.permute(0, 2, 1)  # B, 3, T -> B, T, 3
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # B, 256
        x = self.dropout(x)
        x = torch.cat([x, limb], dim=1)  # B, 256 + limb_dim
        out = self.fc(x)
        return out

# -----------------------------
# Load model
# -----------------------------
model = CNNLimb(num_classes=NUM_CLASSES, limb_dim=LIMB_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# Load limb encoder
# -----------------------------
meta = pd.read_csv("train.csv")
activity_encoder = SimpleLabelEncoder(meta["activity"].unique())
limb_encoder = SimpleLabelEncoder(meta["body_part"].unique())

# -----------------------------
# Load preprocessed test data with limb features
# -----------------------------
def load_chunks_with_limb(chunk_dir, limb_encoder):
    X_list, limb_list, sample_ids = [], [], []
    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".h5"):
            continue
        with h5py.File(os.path.join(chunk_dir, fname), "r") as f:
            for key in f.keys():
                grp = f[key]
                ts = grp["ts"][:]
                sample_id = grp.attrs["sample_id"]
                limb = grp.attrs["body_part"]
                # one-hot encode limb
                limb_onehot = np.zeros(len(limb_encoder.classes_), dtype=np.float32)
                limb_onehot[limb_encoder.transform([limb])[0]] = 1
                X_list.append(ts)
                limb_list.append(limb_onehot)
                sample_ids.append(sample_id)
    return X_list, limb_list, sample_ids

X_test, limb_test, sample_ids = load_chunks_with_limb(CHUNK_DIR, limb_encoder)
test_dataset = TSDataset(X_test, limb_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# Load the label encoder (without fitting)
# -----------------------------
label_encoder = LabelEncoder()
label_encoder.fit(meta["activity"])  # Only fit once during training
print("Label: Encoding")
for label, encoding in activity_encoder.label_dict.items():
    print(f"{label}: {encoding}")

# -----------------------------
# Predict
# -----------------------------
all_preds = []
all_probs = []  # For storing the probabilities

# Checking for NaNs in input data
with torch.no_grad():
    for i, (x_batch, limb_batch) in enumerate(tqdm(test_loader, desc="Predicting")):
        # Check if there are NaNs in the batch
        if torch.isnan(x_batch).any() or torch.isnan(limb_batch).any():
            problematic_sample_ids = [sample_ids[idx] for idx in range(len(x_batch)) if torch.isnan(x_batch[idx]).any() or torch.isnan(limb_batch[idx]).any()]
            print(f"NaN values found in batch {i}, sample IDs: {problematic_sample_ids}")
            continue  # Skip this batch, or handle it as needed

        x_batch = x_batch.to(DEVICE)
        limb_batch = limb_batch.to(DEVICE)
        logits = model(x_batch, limb_batch)

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)  # Store probabilities

        # Get predicted class (with highest probability)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)

# Ensure the probabilities are in float64 format (no scientific notation)
all_probs = np.array(all_probs, dtype=np.float64)

# Ensure probabilities sum to 1
all_probs /= all_probs.sum(axis=1, keepdims=True)

# Final check
print(f"Probability sum (after normalization): {np.sum(all_probs, axis=1)}")

# Save to CSV with fixed-point notation (6 decimals) and avoid scientific notation
submission = pd.DataFrame(all_probs, columns=label_encoder.classes_)
submission["sample_id"] = sample_ids
submission = submission[["sample_id"] + list(label_encoder.classes_)]
submission = submission.fillna(0)  # Replace NaNs with 0 (or another value)

# Save with float_format to prevent scientific notation
submission.to_csv(SUBMISSION_PATH, index=False, float_format='%.6f')  # Ensure no scientific notation in CSV

print(f"Saved predictions to {SUBMISSION_PATH}")
