import pandas as pd
import numpy as np

# Load the submission CSV
df_sub = pd.read_csv('submission.csv')

# 1. Check for any NaNs (null values) in the DataFrame
if df_sub.isnull().values.any():
    print("Found NaNs (null values) in the submission.")
    # Find rows with NaNs
    print(df_sub[df_sub.isnull().any(axis=1)])
else:
    print("No NaNs (null values) in the submission.")

# 2. Check for rows where the sum of the probabilities is not equal to 1
# We will ignore the first column (sample_id) and check the sum of all probability columns
prob_columns = df_sub.columns[1:]  # All columns except "sample_id"

# Check if the sum of probabilities equals 1 for each row
if np.allclose(df_sub[prob_columns].sum(axis=1), 1):
    print("All rows have probabilities summing to 1.")
else:
    print("Some rows do not have probabilities summing to 1.")
    # Print rows where the sum is not 1
    invalid_rows = df_sub[np.abs(df_sub[prob_columns].sum(axis=1) - 1) > 1e-6]
    print(invalid_rows)

# Optionally, you can save the cleaned version of the CSV if needed
# df_sub.to_csv("cleaned_submission.csv", index=False)
