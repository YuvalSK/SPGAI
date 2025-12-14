import pandas as pd

datasets = ['Danenberg', "Jackson"]
d_markers = []
j_markers = []

for d in datasets:
    path = f"Results/{d}/panel.parquet"
    df = pd.read_parquet(path, engine="fastparquet")

    targets = (
        df.sort_index()["target"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(f"Dataset: {d}\n {targets}")
    if d == "Danenberg":
        d_markers = targets
    else:
        j_markers = targets
shared = sorted(set(d_markers) & set(j_markers))
print(len(shared), "shared markers:")
print(shared)



