# Full EDA script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load metadata
metadata = pd.read_csv("metadata.csv")

print("Total rows in metadata:", len(metadata))

print(metadata.isnull().sum())

print("Number of unique sample_ids:", metadata['sample_id'].nunique())

# Count by sensor
sensor_counts = metadata['sensor'].value_counts()
print("Number of samples per sensor:")
print(sensor_counts, "\n")

# Count by body part
body_counts = metadata['body_part'].value_counts()
print("Number of samples per body part:")
print(body_counts, "\n")



# ------------------------------
# 1. Load train.csv
# ------------------------------
train_df = pd.read_csv("train.csv")
print("Total rows in train:", len(train_df))

print(train_df.isnull().sum())

print("Number of unique sample_ids:", train_df['sample_id'].nunique())

# Count by sensor
sensor_counts = train_df['sensor'].value_counts()
print("Number of samples per sensor:")
print(sensor_counts, "\n")

# Count by body part
body_counts = train_df['body_part'].value_counts()
print("Number of samples per body part:")
print(body_counts, "\n")


print("First 5 rows of train.csv:")
print(train_df.head(), "\n")

print("Data info:")
print(train_df.info(), "\n")

print("Basic stats for numeric columns:")
print(train_df.describe(), "\n")

print("Number of samples:", len(train_df))

# Check missing values
print("Missing values:\n", train_df.isnull().sum(), "\n")

# ------------------------------
# 2. Analyze labels
# ------------------------------
if 'activity' in train_df.columns:
    print("Label distribution (activity):\n", train_df['activity'].value_counts())
    plt.figure(figsize=(12, 6))
    sns.countplot(x='activity', data=train_df, order=train_df['activity'].value_counts().index)
    plt.title("Activity Label Distribution")
    plt.xticks(rotation=45)
    plt.show()

# ------------------------------
# 3. Numeric correlations
# ------------------------------
numeric_cols = train_df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlations (Numeric Only)")
plt.show()

# ------------------------------
# 4. Load and explore time series
# ------------------------------
ts_files = [f for f in os.listdir("unlabeled/") if f.endswith('.csv')]
print(f"Number of CSV files in unlabeled/: {len(ts_files)}\n")


# Function to plot a single TS
def plot_ts(ts_file, steps=None, smooth_window=None):
    ts = pd.read_csv(f"unlabeled/{ts_file}")
    print(f"Preview of {ts_file}:")
    print(ts.head(), "\n")

    # Only numeric columns
    ts_numeric = ts[['x', 'y', 'z']]

    # Apply smoothing if requested
    if smooth_window:
        ts_numeric = ts_numeric.rolling(window=smooth_window, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    if steps:
        plt.plot(ts_numeric['x'][:steps], label='x')
        plt.plot(ts_numeric['y'][:steps], label='y')
        plt.plot(ts_numeric['z'][:steps], label='z')
    else:
        plt.plot(ts_numeric['x'], label='x')
        plt.plot(ts_numeric['y'], label='y')
        plt.plot(ts_numeric['z'], label='z')

    plt.title(f"Time Series Plot: {ts_file}")
    plt.xlabel("Time step")
    plt.ylabel("Sensor value")
    plt.legend()
    plt.show()

    print("Basic stats of this TS:\n", ts_numeric.describe(), "\n")


# --------------------------
# Plotting function
# --------------------------
def plot_ts_sensor(ts_file, sensor_type, steps=200, smooth_window=5):
    ts = pd.read_csv(os.path.join("unlabeled/", str(ts_file)))

    if sensor_type == "smartwatch":
        ts_numeric = ts[ts['measurement type'] == 'acceleration [m/s/s]'][['x', 'y', 'z']]
    else:
        # pick columns that contain 'x', 'y', 'z' (handles 'x [m]' etc.)
        xyz_cols = [c for c in ts.columns if 'x' in c.lower() or 'y' in c.lower() or 'z' in c.lower()]
        ts_numeric = ts[xyz_cols]

    # Smooth
    ts_numeric = ts_numeric.rolling(window=smooth_window, min_periods=1).mean()

    # Limit timesteps
    ts_numeric = ts_numeric.iloc[:steps]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(ts_numeric.iloc[:,0], label='x')
    plt.plot(ts_numeric.iloc[:,1], label='y')
    plt.plot(ts_numeric.iloc[:,2], label='z')
    plt.title(f"{sensor_type} Time Series: {ts_file}")
    plt.xlabel("Time step")
    plt.ylabel("Sensor value")
    plt.legend()
    plt.show()

# Plot first 3 TS files for example
for ts_file in ts_files[:3]:
    plot_ts(ts_file, steps=100, smooth_window=5)

# ------------------------------
# 5. Self-supervised tasks suggestions
# ------------------------------
self_supervised_tasks = [
    "1. Time series forecasting: Predict next timesteps from previous timesteps to pretrain the model.",
    "2. Contrastive learning: Generate augmented TS (noise, scaling, rotation) and train model to identify positive/negative pairs."
]

print("Self-supervised tasks for pretraining:")
for task in self_supervised_tasks:
    print(task)


smartwatch_sample = train_df[train_df['sensor'] == 'smartwatch'].iloc[0]['sample_id']
vicon_sample = train_df[train_df['sensor'] == 'vicon'].iloc[0]['sample_id']
smartwatch_file = f"{smartwatch_sample}.csv"
vicon_file = f"{vicon_sample}.csv"

plot_ts_sensor(smartwatch_file, sensor_type="smartwatch")
plot_ts_sensor(vicon_file, sensor_type="vicon")