import os
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

# Directories
RAW_DIR = "unlabeled/"
CHUNK_DIR = "preprocessed_train_chunks50/"
os.makedirs(CHUNK_DIR, exist_ok=True)

# Load metadata
metadata = pd.read_csv("train.csv")

# Function to canonicalize side
def canonicalize(ts, side):
    if side.lower() == "left":
        ts[:, 0] *= -1
    return ts

# Function to compute acceleration from positions
def position_to_acceleration(ts_positions, dt=1.0):
    velocity = np.diff(ts_positions, axis=0) / dt
    velocity = np.vstack([velocity[0:1, :], velocity])
    acceleration = np.diff(velocity, axis=0) / dt
    acceleration = np.vstack([acceleration[0:1, :], acceleration])
    return acceleration

# Chunk parameters
CHUNK_SIZE = 50000
chunk_data = []
chunk_meta = []
chunk_idx = 0

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
    ts_file = os.path.join(RAW_DIR, f"{row['sample_id']}.csv")
    ts = pd.read_csv(ts_file)

    # Check for NaNs in raw data and fill if necessary
    if ts.isnull().values.any():
        print(f"NaNs found in raw data of sample {row['sample_id']} (CSV file).")
        ts = ts.ffill()

    # Smartwatch: Process accelerometer data
    if row['sensor'].lower() == "smartwatch":
        ts = ts[ts['measurement type'] == 'acceleration [m/s/s]']
        ts_numeric = ts[['x', 'y', 'z']].to_numpy()

        # Check if accelerometer data has NaNs and fill
        if np.any(np.isnan(ts_numeric)):
            print(f"NaNs found in accelerometer data of sample {row['sample_id']} (smartwatch).")
            ts_numeric = pd.DataFrame(ts_numeric).fillna(method='ffill').to_numpy()

    else:  # Vicon: compute acceleration from positions
        vicon_cols = [c for c in ts.columns if 'x' in c.lower() or 'y' in c.lower() or 'z' in c.lower()]
        ts_numeric = ts[vicon_cols].to_numpy()

        # Fill NaNs in Vicon data if any
        if np.any(np.isnan(ts_numeric)):
            print(f"NaNs found in Vicon data of sample {row['sample_id']} before acceleration computation.")
            ts_numeric = pd.DataFrame(ts_numeric).fillna(method='ffill').to_numpy()

        # Compute acceleration
        ts_numeric = position_to_acceleration(ts_numeric)

        # Check if computed acceleration contains NaNs and fill
        if np.any(np.isnan(ts_numeric)):
            print(f"NaNs found in computed acceleration of sample {row['sample_id']}.")
            ts_numeric = pd.DataFrame(ts_numeric).fillna(method='ffill').to_numpy()

    # Canonicalize side
    ts_numeric = canonicalize(ts_numeric, row['side'])

    # Check for NaNs after canonicalization and fill
    if np.any(np.isnan(ts_numeric)):
        print(f"NaNs found in canonicalized data of sample {row['sample_id']}.")
        ts_numeric = pd.DataFrame(ts_numeric).fillna(method='ffill').to_numpy()

    # Check again if NaNs exist after filling
    if np.any(np.isnan(ts_numeric)):
        print(f"NaNs still found in data of sample {row['sample_id']} after forward fill.")
        continue  # Skip the sample if NaNs are still present

    # Collect metadata for this sample
    meta_info = {
        'sample_id': row['sample_id'],
        'sensor': row['sensor'],
        'body_part': row['body_part'],
        'side': row['side'],
        'sequence_length': ts_numeric.shape[0]
    }

    # Add data to chunk
    chunk_data.append(ts_numeric.astype(np.float32))
    chunk_meta.append(meta_info)

    # Save chunk if reached CHUNK_SIZE
    if len(chunk_data) >= CHUNK_SIZE:
        out_file = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx}.h5")
        with h5py.File(out_file, "w") as f:
            for i, data in enumerate(chunk_data):
                grp = f.create_group(str(i))
                grp.create_dataset("ts", data=data)
                for key, val in chunk_meta[i].items():
                    grp.attrs[key] = val
        chunk_idx += 1
        chunk_data = []
        chunk_meta = []

# Save remaining data
if len(chunk_data) > 0:
    out_file = os.path.join(CHUNK_DIR, f"chunk_{chunk_idx}.h5")
    with h5py.File(out_file, "w") as f:
        for i, data in enumerate(chunk_data):
            grp = f.create_group(str(i))
            grp.create_dataset("ts", data=data)
            for key, val in chunk_meta[i].items():
                grp.attrs[key] = val

print("Preprocessing complete.")
