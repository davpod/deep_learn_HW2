import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_csv("train.csv")

# We only need the label
y = df["activity"].values

y_train, y_val = train_test_split(
    y,
    test_size=0.2,
    random_state=42,
    stratify=y  # important for balanced classes
)
classes = np.unique(y_train)
K = len(classes)

y_train_pred = np.random.choice(classes, size=len(y_train))
y_val_pred = np.random.choice(classes, size=len(y_val))
# Uniform probabilities for all samples
y_train_proba = np.full((len(y_train), K), 1.0 / K)
y_val_proba = np.full((len(y_val), K), 1.0 / K)

print("Naïve uniform random baseline")

print(f"Train accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Val accuracy:   {accuracy_score(y_val, y_val_pred):.4f}")

print(f"Train log loss: {log_loss(y_train, y_train_proba):.4f}")
print(f"Val log loss:   {log_loss(y_val, y_val_proba):.4f}")

print("\nValidation report:")
print(classification_report(y_val, y_val_pred))
