import pandas as pd

ts_file = "unlabeled/0.csv"
ts = pd.read_csv(ts_file, index_col=False)  # make sure first column is not treated as index
ts.columns = ts.columns.str.strip()          # remove any spaces
print(ts.columns)
print(ts.head())
