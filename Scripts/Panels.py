import pandas as pd

markers = pd.read_parquet("Results/Jackson/panel.parquet", engine="pyarrow")
print(markers.head())