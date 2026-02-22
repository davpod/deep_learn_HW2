# ============================================================
# InceptionTime + Handcrafted Features → XGBoost
# ============================================================

import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch import amp
from xgboost import plot_importance
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA

import xgboost as xgb

# tsai
from tsai.models.InceptionTime import InceptionTime

# ============================================================
# SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16 = True
BATCH_SIZE = 128

# ============================================================
# METADATA
# ============================================================

### CHANGE HERE ###
TRAIN_META_PATH = "train.csv"
TEST_META_PATH  = "metadata.csv"

train_meta = pd.read_csv(TRAIN_META_PATH)
test_meta  = pd.read_csv(TEST_META_PATH)

label_map = dict(zip(train_meta["sample_id"], train_meta["activity"]))
limb_map  = dict(zip(train_meta["sample_id"], train_meta["body_part"]))
limb_map_test = dict(zip(test_meta["sample_id"], test_meta["body_part"]))

le = LabelEncoder()
le.fit(train_meta["activity"])

# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(chunk_dir, label_map=None):
    X_list, y_list, sample_ids = [], [], []

    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".h5"):
            continue

        with h5py.File(os.path.join(chunk_dir, fname), "r") as f:
            for key in f.keys():
                grp = f[key]
                ts = grp["ts"][:]

                sample_id = grp.attrs["sample_id"]
                if isinstance(sample_id, bytes):
                    sample_id = sample_id.decode("utf-8")

                if label_map is not None:
                    if sample_id not in label_map:
                        continue
                    y_list.append(label_map[sample_id])

                X_list.append(ts)
                sample_ids.append(sample_id)

    return X_list, y_list, sample_ids

### CHANGE HERE ###
TRAIN_CHUNKS_DIR = "preprocessed_train_chunks50/"
TEST_CHUNKS_DIR  = "preprocessed_chunks50/"

X_list, y_raw, _ = load_chunks(TRAIN_CHUNKS_DIR, label_map)
X_test_list, _, test_ids = load_chunks(TEST_CHUNKS_DIR)
print("loaded test data")
y = le.transform(y_raw)

# ============================================================
# HANDCRAFTED FEATURES
# ============================================================

def extract_handcrafted_features(X_list):
    feats = []

    for X in X_list:
        f = []

        for axis in range(3):
            x = X[:, axis]
            f.extend([
                x.mean(),
                x.std(),
                x.min(),
                x.max(),
                np.ptp(x),
                np.mean(np.abs(x)),
                skew(x),
                kurtosis(x),
            ])

        mag = np.linalg.norm(X, axis=1)
        f.extend([
            mag.mean(),
            mag.std(),
            mag.max(),
            np.mean(mag ** 2),
            len(X)
        ])

        feats.append(f)

    return np.array(feats)

print("Extracting handcrafted features...")
X_hand = extract_handcrafted_features(X_list)
X_test_hand = extract_handcrafted_features(X_test_list)

# ============================================================
# INCEPTION FEATURE EXTRACTOR
# ============================================================

class InceptionTimeFeatureExtractor(nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        self.backbone = InceptionTime(c_in=c_in, c_out=1)
        self.backbone.fc = nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, c_in, 100)
            self.feat_dim = self.backbone(dummy).shape[1]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.backbone(x)


extractor = InceptionTimeFeatureExtractor(c_in=3).to(DEVICE)

### CHANGE HERE ###
CHECKPOINT_PATH = "best_inceptiontime_limbl21_120.pth"

extractor.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE),
    strict=False
)

extractor.eval()
for p in extractor.parameters():
    p.requires_grad = False

@torch.no_grad()
def extract_dl_features(model, X_list):
    feats_all = []

    for i in range(0, len(X_list), BATCH_SIZE):
        xs = [torch.tensor(x, dtype=torch.float32) for x in X_list[i:i+BATCH_SIZE]]
        xs = pad_sequence(xs, batch_first=True).to(DEVICE)

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=BF16):
            feats = model(xs)

        feats_all.append(feats.float().cpu().numpy())

    return np.vstack(feats_all)

print("Extracting deep features...")
X_dl = extract_dl_features(extractor, X_list)
X_test_dl = extract_dl_features(extractor, X_test_list)

# ============================================================
# CONCATENATE FEATURES
# ============================================================

X_full = np.hstack([X_hand, X_dl])
X_test_full = np.hstack([X_test_hand, X_test_dl])

# ============================================================
# TRAIN / VAL SPLIT (SEQUENTIAL)
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# SCALE + PCA (OPTION 1: AUTO)
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test_full = scaler.transform(X_test_full)

# PCA automatically adapts to feature size
n_features = X_train.shape[1]
pca = PCA(n_components=min(128, n_features), random_state=42)

X_train = pca.fit_transform(X_train)
X_val   = pca.transform(X_val)
X_test_full = pca.transform(X_test_full)

# ============================================================
# XGBOOST (BEST TREE FOR EMBEDDINGS)
# ============================================================

model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,            # shallower trees
    learning_rate=0.03,     # slower learning
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    tree_method="hist", # ensure GPU
    device="cuda",
    random_state=42,
    n_jobs=-1
)
print("Training model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# ============================================================
# EVALUATION
# ============================================================

train_pred = model.predict(X_train)
train_proba = model.predict_proba(X_train)

val_pred = model.predict(X_val)
val_proba = model.predict_proba(X_val)

print("Train accuracy:", accuracy_score(y_train, train_pred))
print("Train log loss:", log_loss(y_train, train_proba))
print("Val accuracy:", accuracy_score(y_val, val_pred))
print("Val log loss:", log_loss(y_val, val_proba))

# ============================================================
# TEST → submission.csv
# ============================================================

y_test_proba = model.predict_proba(X_test_full)

df_sub = pd.DataFrame(y_test_proba, columns=le.classes_)
df_sub.insert(0, "sample_id", test_ids)
df_sub.to_csv("submission.csv", index=False)

print("submission.csv saved")
print("Probabilities sum to 1:",
      np.allclose(df_sub.iloc[:, 1:].sum(axis=1), 1.0))
# Plot importance by default (gain)
plot_importance(model, importance_type='gain', max_num_features=20)
plt.show()