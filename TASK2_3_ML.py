import os
import h5py
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
import joblib

def load_chunks(chunk_dir, label_map):
    """
    label_map: dict sample_id -> activity label
    """
    X_list = []
    y_list = []

    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".h5"):
            continue

        with h5py.File(os.path.join(chunk_dir, fname), "r") as f:
            for key in f.keys():
                grp = f[key]

                ts = grp["ts"][:]          # (T, 3)
                sample_id = grp.attrs["sample_id"]

                # only keep labeled samples
                if sample_id not in label_map:
                    continue

                X_list.append(ts)
                y_list.append(label_map[sample_id])

    return X_list, np.array(y_list)
def load_chunks_with_limb(chunk_dir, label_map, limb_map):
    """
    Returns X_list, y_list, limb_list
    """
    X_list = []
    y_list = []
    limb_list = []

    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".h5"):
            continue

        with h5py.File(os.path.join(chunk_dir, fname), "r") as f:
            for key in f.keys():
                grp = f[key]
                ts = grp["ts"][:]
                sample_id = grp.attrs["sample_id"]

                if sample_id not in label_map:
                    continue

                X_list.append(ts)
                y_list.append(label_map[sample_id])

                # Limb feature
                limb = limb_map[sample_id]  # 'hand' or 'foot'
                limb_list.append(limb)

    return X_list, np.array(y_list), limb_list

def extract_features_varlen(X_list):
    """
    X_list: list of arrays, each (T_i, 3)
    returns: (N, F)
    """
    features = []

    for X in X_list:
        feats = []

        # per-axis stats
        for axis in range(3):
            x = X[:, axis]

            feats.extend([
                x.mean(),
                x.std(),
                x.min(),
                x.max(),
                np.ptp(x),
                np.mean(np.abs(x)),
                skew(x),
                kurtosis(x),
            ])

        # vector magnitude
        mag = np.linalg.norm(X, axis=1)
        feats.extend([
            mag.mean(),
            mag.std(),
            mag.max(),
            np.mean(mag ** 2),  # energy
        ])

        # length as feature
        feats.append(len(X))

        features.append(feats)

    return np.array(features)

meta = pd.read_csv("train.csv")
label_map = dict(zip(meta["sample_id"], meta["activity"]))
limb_map = dict(zip(meta["sample_id"], meta["body_part"]))

le = LabelEncoder()
le.fit(meta["activity"])

X_list, y_raw = load_chunks("preprocessed_train_chunks50/", label_map)
#X_list,y_raw,limb_list = load_chunks_with_limb("preprocessed_train_chunks50/", label_map,limb_map)
# Create encoder for limb
#limb_encoder = {'hand': 0, 'foot': 1}  # two classes
#limb_onehot = np.zeros((len(limb_list), 2), dtype=np.float32)
#for i, limb in enumerate(limb_list):
#    limb_onehot[i, limb_encoder[limb]] = 1

print("loaded train data")
y = le.transform(y_raw)
X_feat = extract_features_varlen(X_list)
#X_feat_aug = np.hstack([X_feat, limb_onehot])
print("Loaded samples:", len(X_list))
print("Loaded labels:", len(y))
split_index = int(0.8 * len(X_feat))  # 80% train, 20% val
X_train, X_val = X_feat[:split_index], X_feat[split_index:]
y_train, y_val = y[:split_index], y[split_index:]
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=7,  # each leaf must have ≥5 samples
        random_state=42,
        n_jobs=-1
    ))
])
"""
train accuracy: 0.7908602418030748
train log loss: 0.8945118357941908
Validation accuracy: 0.7653731343283582
Validation log loss: 0.9417825493162258
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,  # each leaf must have ≥5 samples
        random_state=42,
        n_jobs=-1
test score: 1.49919
"""
'''
train accuracy: 0.8610129857206826
train log loss: 0.6969428122742956
Validation accuracy: 0.8249751243781095
Validation log loss: 0.7652482931903246
n_estimators=300,
        max_depth=12,
        min_samples_leaf=6,  # each leaf must have ≥5 samples
        random_state=42,
        n_jobs=-1
score: 1.44537 ---------------------------------------best

train accuracy: 0.9351957808846211
train log loss: 0.4883165705511067
Validation accuracy: 0.8927363184079602
Validation log loss: 0.582488246897764
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=7,  # each leaf must have ≥5 samples
        random_state=42,
        n_jobs=-1
score: 1.40447
'''
"""
train accuracy: 0.6450072142892681
train log loss: 1.2885731543960361
Validation accuracy: 0.6381094527363184
Validation log loss: 1.3083470330162557
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=5,  # each leaf must have ≥5 samples
        random_state=42,
        n_jobs=-1
score 1.65762
"""
print("entering training")
pipe.fit(X_train, y_train)

y_train_pred = pipe.predict(X_train)
y_train_proba = pipe.predict_proba(X_train)
y_val_pred = pipe.predict(X_val)
y_val_proba = pipe.predict_proba(X_val)

print("train accuracy:", accuracy_score(y_train, y_train_pred))
print("train log loss:", log_loss(y_train, y_train_proba))
print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
print("Validation log loss:", log_loss(y_val, y_val_proba))

# Save the trained model
joblib.dump(pipe, "rf_model.pkl")

# Later, load it
pipe = joblib.load("rf_model.pkl")

def load_test_chunks(chunk_dir):
    X_list = []
    sample_ids = []

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

                X_list.append(ts)
                sample_ids.append(sample_id)

    return X_list, sample_ids
def load_test_chunks_with_limb(chunk_dir, limb_map):
    X_list = []
    sample_ids = []
    limb_list = []

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

                X_list.append(ts)
                sample_ids.append(sample_id)

                limb = limb_map[sample_id]  # test limb map
                limb_list.append(limb)

    return X_list, sample_ids, limb_list
test_meta = pd.read_csv("metadata.csv")  # if you have one
limb_map_test = dict(zip(test_meta["sample_id"], test_meta["body_part"]))
print("entering evaluation")
# ---- create one-hot for test ----
X_test_list, sample_ids = load_test_chunks("preprocessed_chunks50/")
print("loaded test data")
X_test_feat = extract_features_varlen(X_test_list)

#limb_onehot_test = np.zeros((len(limb_list_test), 2), dtype=np.float32)
#for i, limb in enumerate(limb_list_test):
#    limb_onehot_test[i, limb_encoder[limb]] = 1

#X_test_feat_aug = np.hstack([X_test_feat, limb_onehot_test])
y_test_proba = pipe.predict_proba(X_test_feat)

class_names = le.classes_
df_sub = pd.DataFrame(y_test_proba, columns=class_names)
df_sub.insert(0, "sample_id", sample_ids)
df_sub.to_csv("submission.csv", index=False)
print(np.allclose(df_sub.iloc[:, 1:].sum(axis=1), 1))
