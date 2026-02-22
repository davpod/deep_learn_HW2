import os
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import amp
from tsai.models.InceptionTime import InceptionTime

# -----------------------------
# Settings
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1000
CHUNK_DIR = "preprocessed_chunks50/"
MODEL_PATH = "best_inceptiontime_limbl19_124.pth"
SUBMISSION_PATH = "submission.csv"
NUM_CLASSES = 18
BF16 = True
SMALL_PROB_THRESHOLD = 1e-5


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

    def inverse_transform(self, encoded_labels):
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
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    return xs_padded, limbs


# -----------------------------
# Model Definition
# -----------------------------
class InceptionTimeLimb(nn.Module):
    def __init__(self, c_in=3, limb_dim=6, num_classes=18, dropout_rate=0.5):
        super().__init__()
        self.backbone = InceptionTime(c_in=c_in, c_out=1)
        self.backbone.fc = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, c_in, 100)
            backbone_out_dim = self.backbone(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(backbone_out_dim + limb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, limb):
        x = x.permute(0, 2, 1)
        feats = self.backbone(x)
        feats = torch.cat([feats, limb], dim=1)
        return self.classifier(feats)


# -----------------------------
# Load metadata and encoders
# -----------------------------
meta = pd.read_csv("train.csv")
activity_encoder = SimpleLabelEncoder(meta["activity"].unique())
limb_encoder = SimpleLabelEncoder(meta["body_part"].unique())

# -----------------------------
# Load model
# -----------------------------
model = InceptionTimeLimb(
    c_in=3,
    limb_dim=len(limb_encoder.classes_),
    num_classes=NUM_CLASSES,
    dropout_rate=0.5
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# -----------------------------
# Load test chunks
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
# Predict
# -----------------------------
all_probs = []
all_preds = []

with torch.no_grad():
    for i, (x_batch, limb_batch) in enumerate(tqdm(test_loader, desc="Predicting")):
        x_batch = x_batch.to(DEVICE)
        limb_batch = limb_batch.to(DEVICE)

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=BF16):
            logits = model(x_batch, limb_batch)

        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(np.argmax(probs, axis=1))

# -----------------------------
# Handle small probabilities
# -----------------------------
all_probs = np.array(all_probs)
small_prob_mask = all_probs < SMALL_PROB_THRESHOLD
all_probs[small_prob_mask] = 0.0

for i in range(all_probs.shape[0]):
    mass_to_redistribute = SMALL_PROB_THRESHOLD * small_prob_mask[i].sum()
    if mass_to_redistribute > 0:
        most_confident_idx = np.argmax(all_probs[i])
        all_probs[i, most_confident_idx] += mass_to_redistribute

all_probs /= all_probs.sum(axis=1, keepdims=True)

# -----------------------------
# Save submission
# -----------------------------
submission = pd.DataFrame(all_probs, columns=activity_encoder.classes_)
submission["sample_id"] = sample_ids
submission = submission[["sample_id"] + list(activity_encoder.classes_)]
submission.to_csv(SUBMISSION_PATH, index=False, float_format="%.6f")

print(f"Saved predictions to {SUBMISSION_PATH}")
