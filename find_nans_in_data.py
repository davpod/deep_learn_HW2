import numpy as np
import h5py
import os


def check_for_nans_in_chunks(chunk_dir, batch_size=100):
    """
    Check for NaN values in the chunks before passing to the model.

    Parameters:
    - chunk_dir (str): Directory where the chunks are stored
    - batch_size (int): Number of samples to load at once for the check

    Returns:
    - None: Prints the chunks and batches with NaNs found
    """
    # Loop through each chunk file
    for fname in sorted(os.listdir(chunk_dir)):
        if not fname.endswith(".h5"):
            continue

        with h5py.File(os.path.join(chunk_dir, fname), "r") as f:
            # Check each sample in the chunk
            for key in f.keys():
                grp = f[key]
                ts = grp["ts"][:]  # time series data

                # Check if there are any NaN values in the time series data
                if np.any(np.isnan(ts)):
                    print(f"NaNs found in chunk {fname}, sample {key}")
                    print(f"Sample data with NaNs:\n{ts[np.isnan(ts)]}")

                # Also check if the limb feature exists and is valid (not NaN or empty)
                limb = grp.attrs["body_part"]
                if limb is None or limb == "":
                    print(f"Invalid limb feature for sample {key} in chunk {fname}: {limb}")

            print(f"Processed chunk {fname} - No NaNs found in time series.")
check_for_nans_in_chunks("preprocessed_chunks50/",1000)