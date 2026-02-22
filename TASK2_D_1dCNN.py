import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import amp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
from sklearn.utils.class_weight import compute_class_weight

start_time = time.time()

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
BATCH_SIZE = 256        # batch size for training
NUM_EPOCHS = 100           # number of training epochs
LEARNING_RATE = 2e-4      # Adam optimizer learning rate
DROPOUT_RATE = 0.3        # dropout in CNN
NUM_CLASSES = 18          # number of activity classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = True               # use bfloat16 mixed precision

# -----------------------------
# Custom Encoder for Consistent Label Encoding
# -----------------------------
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
    def __init__(self, X_list, limb_list, y_list=None, augment=False):
        self.X_list = X_list
        self.limb_list = limb_list
        self.y_list = y_list
        self.augment = augment  # Store whether augmentation is enabled

    def __len__(self):
        return len(self.X_list)

    def jittering(self, X, strength=0.02):
        noise = np.random.normal(0, strength, X.shape)  # Add noise with mean 0 and standard deviation `strength`
        return X + noise

    def scaling(self, X):
        scale_factor = np.random.uniform(0.8, 1.2)  # Random scale factor
        return X * scale_factor

    def time_warping(self, X):
        warp_factor = np.random.uniform(0.8, 1.2)
        length = len(X)
        new_length = int(length * warp_factor)

        # Apply time warping separately to each axis (dimension) of X
        X_warped = np.zeros((new_length, X.shape[1]))  # Initialize the warped array

        for i in range(X.shape[1]):  # Iterate through the 3 axes (x, y, z)
            X_warped[:, i] = np.interp(np.arange(0, new_length), np.arange(0, length), X[:, i])

        return X_warped

    def random_shift(self, X, shift_range=20):
        shift = np.random.randint(-shift_range, shift_range)
        return np.roll(X, shift)

    def augment_data(self, X):
        # Apply augmentations
        X = self.jittering(X)  # Jittering (Gaussian noise)
        X = self.scaling(X)  # Scaling (Random scaling)
        X = self.time_warping(X)  # Time warping
        X = self.random_shift(X)  # Random shifting
        return torch.tensor(X, dtype=torch.float32)  # Ensure it's a tensor

    def __getitem__(self, idx):
        x = np.array(self.X_list[idx], dtype=np.float32)  # (T, 3)
        limb = np.array(self.limb_list[idx], dtype=np.float32)

        if self.augment:  # Apply augmentations only if `augment=True`
            x = self.augment_data(x)

        if self.y_list is not None:
            y = np.array(self.y_list[idx], dtype=np.int64)  # Change to np.int64 or torch.long
            return torch.tensor(x).clone(), torch.tensor(limb).clone(), torch.tensor(y).clone()
        else:
            return torch.tensor(x).clone(), torch.tensor(limb).clone()

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
# Model
# -----------------------------
class CNNLimb(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, limb_dim=6):
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
        x = x.permute(0, 2, 1)  # (B, 3, T)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (B, 256)
        x = self.dropout(x)
        x = torch.cat([x, limb], dim=1)  # (B, 256 + limb_dim)
        out = self.fc(x)
        return out

# -----------------------------
# Train / Validate functions
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct, total = 0, 0
    loss_accum = 0

    for imgs, limb, lbls in loader:
        imgs, limb, lbls = imgs.to(device), limb.to(device), lbls.to(device)

        # Cast input to bfloat16 for mixed precision
        imgs = imgs.to(dtype=torch.bfloat16)

        optimizer.zero_grad()

        with amp.autocast(device_type='cuda', dtype=torch.bfloat16):
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

            # Cast input to bfloat16 for mixed precision
            imgs = imgs.to(dtype=torch.bfloat16)

            with amp.autocast(device_type='cuda', dtype=torch.bfloat16):
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
# Load data
# -----------------------------
meta = pd.read_csv("train.csv")
activity_encoder = SimpleLabelEncoder(meta["activity"].unique())
limb_encoder = SimpleLabelEncoder(meta["body_part"].unique())
print("Label: Encoding")
for label, encoding in activity_encoder.label_dict.items():
    print(f"{label}: {encoding}")
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

X_list, y_list, limb_list = load_chunks_with_limb("preprocessed_train_chunks50/", label_map, limb_map)
print("loaded train set")

# -----------------------------
# Compute Class Weights
# -----------------------------
class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_list), y=y_list
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

# -----------------------------
# Sequential Split for Time-Series Data
# -----------------------------
split_index = int(0.8 * len(X_list))  # 80% for training, 20% for validation

X_train, X_val = X_list[:split_index], X_list[split_index:]
y_train, y_val = y_list[:split_index], y_list[split_index:]
limb_train, limb_val = limb_list[:split_index], limb_list[split_index:]

# Train and validation datasets
train_dataset = TSDataset(X_train, limb_train, y_train,augment=True)
val_dataset = TSDataset(X_val, limb_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# Model, optimizer, scaler
# -----------------------------
model = CNNLimb(num_classes=NUM_CLASSES, limb_dim=limb_train.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -----------------------------
# Training loop
# -----------------------------
train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_val_loss = float('inf')
patience = 5  # stop if no improvement in N epochs
counter = 0
for epoch in range(NUM_EPOCHS):
    # train
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # validate
    val_loss, val_acc, val_probs, val_labels = validate_one_epoch(model, val_loader, criterion, DEVICE)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | "
          f"Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    # Check for best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  --> Best model saved with Val Loss {best_val_loss:.4f}")
        counter = 0  # reset patience counter
    else:
        counter += 1
        print(f"  --> No improvement for {counter} epochs")

    # Early stopping
    if counter >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement.")
        break

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), "cnn_limb_bf16.pth")

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
