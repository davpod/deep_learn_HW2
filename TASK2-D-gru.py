import os
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import time

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32  # Smaller batch size for faster convergence
NUM_EPOCHS = 20  # Reduced epochs for quick overfitting check
LEARNING_RATE = 1e-3  # Increased learning rate for faster training
NUM_CLASSES = 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT_RATE = 0.3
BF16 = True

start_time = time.time()

# -----------------------------
# Simple Label Encoder Class
# -----------------------------
class SimpleLabelEncoder:
    def __init__(self, labels):
        self.classes_ = sorted(labels)  # Sort and assign encoding
        self.label_dict = {label: idx for idx, label in enumerate(self.classes_)}
        self.inv_label_dict = {idx: label for label, idx in self.label_dict.items()}

    def transform(self, labels):
        return [self.label_dict[label] for label in labels]

    def inverse_transform(self, encoded_labels):
        return [self.inv_label_dict[encoded] for encoded in encoded_labels]

# -----------------------------
# Dataset Class
# -----------------------------
class TSDataset(Dataset):
    def __init__(self, X_list, limb_list, y_list=None):
        self.X_list = X_list
        self.limb_list = limb_list
        self.y_list = y_list

    def __len__(self):
        return len(self.X_list)

    def __getitem__(self, idx):
        x = torch.tensor(self.X_list[idx], dtype=torch.float32)  # (T, 3)
        limb = torch.tensor(self.limb_list[idx], dtype=torch.float32)
        if self.y_list is not None:
            y = torch.tensor(self.y_list[idx], dtype=torch.long)
            return x, limb, y
        else:
            return x, limb

def collate_fn(batch):
    xs = [item[0] for item in batch]
    limbs = torch.stack([item[1] for item in batch])
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    if len(batch[0]) == 3:
        ys = torch.tensor([item[2] for item in batch])
        return xs_padded, limbs, ys
    else:
        return xs_padded, limbs

# -----------------------------
# GRU Model
# -----------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_classes=NUM_CLASSES, limb_dim=6):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=0.3, bidirectional=True)  # Bidirectional GRU
        self.fc1 = nn.Linear(hidden_size * 2 + limb_dim, 256)  # Adjusted for bidirectional output
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, limb):
        gru_out, hn = self.gru(x)  # GRU output
        gru_out = gru_out[:, -1, :]  # Take output from the last time step
        x = torch.cat([gru_out, limb], dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -----------------------------
# Training and Validation Functions
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total = 0, 0
    loss_accum = 0

    for imgs, limb, lbls in loader:
        imgs, limb, lbls = imgs.to(device), limb.to(device), lbls.to(device)

        optimizer.zero_grad()

        outputs = model(imgs, limb)
        loss = criterion(outputs, lbls)

        loss.backward()
        optimizer.step()

        loss_accum += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    return loss_accum / total, correct / total

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    loss_accum = 0
    outputs_all, labels_all = [], []

    with torch.no_grad():
        for imgs, limb, lbls in loader:
            imgs, limb, lbls = imgs.to(device), limb.to(device), lbls.to(device)
            outputs = model(imgs, limb)
            loss = criterion(outputs, lbls)

            loss_accum += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            outputs_all.append(F.softmax(outputs, dim=1).cpu())
            labels_all.append(lbls.cpu())

    return (
        loss_accum / total,
        correct / total,
        torch.cat(outputs_all),
        torch.cat(labels_all)
    )

# -----------------------------
# Load Data
# -----------------------------
meta = pd.read_csv("train.csv")
activity_encoder = SimpleLabelEncoder(meta["activity"].unique())
limb_encoder = SimpleLabelEncoder(meta["body_part"].unique())
label_map = dict(zip(meta["sample_id"], meta["activity"]))
limb_map = dict(zip(meta["sample_id"], meta["body_part"]))

def load_chunks_with_limb(chunk_dir, label_map, limb_map):
    X_list, y_list, limb_list = [], [], []
    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".h5"): continue
        with h5py.File(os.path.join(chunk_dir, fname), "r") as f:
            for key in f.keys():
                grp = f[key]
                ts = grp["ts"][:]
                sample_id = grp.attrs["sample_id"]
                limb = grp.attrs["body_part"]
                if sample_id not in label_map: continue
                X_list.append(ts)
                y_list.append(activity_encoder.transform([label_map[sample_id]])[0])
                limb_onehot = np.zeros(len(limb_encoder.classes_), dtype=np.float32)
                limb_onehot[limb_encoder.transform([limb])[0]] = 1
                limb_list.append(limb_onehot)
    return X_list, np.array(y_list), np.array(limb_list)

# Reduce dataset for sanity check (e.g., using only 10 samples)
X_list, y_list, limb_list = load_chunks_with_limb("preprocessed_train_chunks50/", label_map, limb_map)
X_train, X_val, y_train, y_val, limb_train, limb_val = train_test_split(
    X_list[:100], y_list[:100], limb_list[:100], test_size=0.2, random_state=42, stratify=None
)

train_dataset_small = TSDataset(X_train, limb_train, y_train)
val_dataset_small = TSDataset(X_val, limb_val, y_val)
train_loader_small = DataLoader(train_dataset_small, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader_small = DataLoader(val_dataset_small, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# Model, Optimizer, Criterion
# -----------------------------
model = GRUModel(input_size=3, hidden_size=128, num_classes=NUM_CLASSES, limb_dim=limb_train.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Training Loop
# -----------------------------
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_loss = float('inf')
patience = 3  # Lower patience for quicker overfitting detection
counter = 0

for epoch in range(NUM_EPOCHS):
    # Train on small dataset
    train_loss, train_acc = train_one_epoch(model, train_loader_small, optimizer, criterion, DEVICE)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate on small validation set
    val_loss, val_acc, val_probs, val_labels = validate_one_epoch(model, val_loader_small, criterion, DEVICE)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    # Check for best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_gru_model.pth")
        print(f"  --> Best model saved with Val Loss {best_val_loss:.4f}")
        counter = 0  # reset patience counter
    else:
        counter += 1
        print(f"  --> No improvement for {counter} epochs")

    # Early stopping (only if you want to use it)
    if counter >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement.")
        break

# Save model
torch.save(model.state_dict(), "gru_limb_small_dataset.pth")

# -----------------------------
# Visualizations
# -----------------------------
# 1. Loss and Accuracy over epochs
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epochs")
plt.legend()

plt.tight_layout()
plt.show()

# 2. Confusion Matrix for validation set
val_preds = val_probs.argmax(dim=1)
cm = confusion_matrix(val_labels, val_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=activity_encoder.classes_)
plt.figure(figsize=(10,10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Validation Confusion Matrix")
plt.show()

end_time = time.time()
runtime_seconds = end_time - start_time
print(f"Runtime: {runtime_seconds:.2f} seconds")
