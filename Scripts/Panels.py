import pandas as pd
from collections import Counter

datasets = ['Danenberg', "Jackson"]
# 1) Read markers per dataset
markers_by_ds = {}
for ds in datasets:
    path = f"Results/{ds}/panel.parquet"
    df = pd.read_parquet(path, engine="fastparquet")
    df.head()
    markers = (
        df["target"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    # keep deterministic ordering
    markers_by_ds[ds] = sorted(markers)

print(markers_by_ds)
# 2) Build global counts: in how many datasets each marker appears
marker_counts = Counter()
for ds, markers in markers_by_ds.items():
    marker_counts.update(set(markers))  # set() so each dataset counts once per marker

# 3) Print unique + shared-per-dataset
for ds in datasets:
    markers = set(markers_by_ds[ds])

    unique_to_ds = sorted([m for m in markers if marker_counts[m] == 1])
    shared_with_others = sorted([m for m in markers if marker_counts[m] > 1])

    print(f"\n=== {ds} ===")
    print(f"{len(unique_to_ds)} unique markers:")
    print(unique_to_ds)
    print(f"{len(shared_with_others)} shared markers (with at least one other dataset):")
    print(shared_with_others)



