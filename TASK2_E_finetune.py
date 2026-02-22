import os
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import amp

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# tsai
from tsai.models.InceptionTime import InceptionTime

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.5
NUM_CLASSES = 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
BF16 = True

start_time = time.time()

# -----------------------------
# Label Encoder
# -----------------------------
class SimpleLabelEncoder:
    def __init__(self, labels):
        self.classes_ = sorted(labels)
        self.label_dict = {label: idx for idx, label in enumerate(self.classes_)}
        self.inv_label_dict = {idx: label for label, idx in self.label_dict.items()}

    def transform(self, labels):
        return [self.label_dict[label] for label in labels]

# -----------------------------
# Dataset
# -----------------------------
class TSDataset(Dataset):
    def __init__(self, X_list, limb_list, y_list):
        self.X_list = X_list
        self.limb_list = limb_list
        self.y_list = y_list

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_list[idx], dtype=torch.float32)  # (T, 3)
        limb = torch.tensor(self.limb_list[idx], dtype=torch.float32)
        y = torch.tensor(self.y_list[idx], dtype=torch.long)
        return x, limb, y

def collate_fn(batch):
    xs = [item[0] for item in batch]           # (T, 3)
    limbs = torch.stack([item[1] for item in batch])
    ys = torch.tensor([item[2] for item in batch])

    xs_padded = pad_sequence(xs, batch_first=True)  # (B, T, 3)
    return xs_padded, limbs, ys

# -----------------------------
# MODEL: InceptionTime + Limb
# -----------------------------
class InceptionTimeLimb(nn.Module):
    def __init__(self, c_in=3, limb_dim=6, num_classes=18):
        super().__init__()

        # Dummy c_out required by tsai
        self.backbone = InceptionTime(c_in=c_in, c_out=1)

        # Disable internal classifier
        self.backbone.fc = nn.Identity()

        # ---- infer backbone output dim safely ----
        with torch.no_grad():
            dummy = torch.zeros(1, c_in, 100)   # (B, C, T)
            backbone_out_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dim + limb_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, limb):
        x = x.permute(0, 2, 1)        # (B, C, T)
        feats = self.backbone(x)      # (B, backbone_out_dim)
        feats = torch.cat([feats, limb], dim=1)
        return self.classifier(feats)

# -----------------------------
# Load Data
# -----------------------------
meta = pd.read_csv("train.csv")
activity_encoder = SimpleLabelEncoder(meta["activity"].unique())
limb_encoder = SimpleLabelEncoder(meta["body_part"].unique())

label_map = dict(zip(meta["sample_id"], meta["activity"]))
limb_map = dict(zip(meta["sample_id"], meta["body_part"]))

def load_chunks_with_limb(chunk_dir):
    X_list, y_list, limb_list = [], [], []

    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".h5"):
            continue

        with h5py.File(os.path.join(chunk_dir, fname), "r") as f:
            for key in f.keys():
                grp = f[key]
                ts = grp["ts"][:]                     # (T, 3)
                sample_id = grp.attrs["sample_id"]
                limb = grp.attrs["body_part"]

                if sample_id not in label_map:
                    continue

                X_list.append(ts)
                y_list.append(activity_encoder.transform([label_map[sample_id]])[0])

                limb_onehot = np.zeros(len(limb_encoder.classes_), dtype=np.float32)
                limb_onehot[limb_encoder.transform([limb])[0]] = 1
                limb_list.append(limb_onehot)

    return X_list, np.array(y_list), np.array(limb_list)

X_list, y_list, limb_list = load_chunks_with_limb("preprocessed_train_chunks50/")

# -----------------------------
# Sequential Split
# -----------------------------
split_index = int(0.8 * len(X_list))

X_train, X_val = X_list[:split_index], X_list[split_index:]
y_train, y_val = y_list[:split_index], y_list[split_index:]
limb_train, limb_val = limb_list[:split_index], limb_list[split_index:]

train_dataset = TSDataset(X_train, limb_train, y_train)
val_dataset = TSDataset(X_val, limb_val, y_val)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# -----------------------------
# Model / Optimizer
# -----------------------------
model = InceptionTimeLimb(
    c_in=3,
    limb_dim=limb_train.shape[1],
    num_classes=NUM_CLASSES
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Train / Validate
# -----------------------------
def train_one_epoch(model, loader):
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for x, limb, y in loader:
        x, limb, y = x.to(DEVICE), limb.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=BF16):
            out = model(x, limb)
            loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum / total, correct / total

def validate_one_epoch(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    probs_all, labels_all = [], []

    with torch.no_grad():
        for x, limb, y in loader:
            x, limb, y = x.to(DEVICE), limb.to(DEVICE), y.to(DEVICE)

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=BF16):
                out = model(x, limb)
                loss = criterion(out, y)

            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

            probs_all.append(F.softmax(out, dim=1).cpu())
            labels_all.append(y.cpu())

    return loss_sum / total, correct / total, torch.cat(probs_all), torch.cat(labels_all)

# -----------------------------
# Training Loop
# -----------------------------
best_val_loss = float("inf")
patience, counter = 5, 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(NUM_EPOCHS):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    va_loss, va_acc, va_probs, va_labels = validate_one_epoch(model, val_loader)

    train_losses.append(tr_loss)
    val_losses.append(va_loss)
    train_accs.append(tr_acc)
    val_accs.append(va_acc)

    print(f"Epoch {epoch+1}: "
          f"Train Loss {tr_loss:.4f} | Train Acc {tr_acc:.4f} | "
          f"Val Loss {va_loss:.4f} | Val Acc {va_acc:.4f}")

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        torch.save(model.state_dict(), "best_inceptiontime_limb.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# -----------------------------
# Confusion Matrix
# -----------------------------
preds = va_probs.argmax(1)
cm = confusion_matrix(va_labels, preds)

disp = ConfusionMatrixDisplay(cm, display_labels=activity_encoder.classes_)
plt.figure(figsize=(10,10))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Validation Confusion Matrix")
plt.show()

print(f"Runtime: {time.time() - start_time:.1f}s")
