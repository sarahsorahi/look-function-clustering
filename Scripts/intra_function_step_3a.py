import pandas as pd
import os
import json

# =========================
# Config
# =========================

TARGET_FUNC = "DIR"
OUT_DIR = "intra_function_artifacts"

cluster_file = f"{OUT_DIR}/{TARGET_FUNC}_hdbscan_clusters.csv"
color_map_file = f"{OUT_DIR}/{TARGET_FUNC}_cluster_color_map.json"

out_dir = f"{OUT_DIR}/{TARGET_FUNC}_cluster_samples"
os.makedirs(out_dir, exist_ok=True)

# =========================
# Load data
# =========================

df = pd.read_csv(cluster_file)

with open(color_map_file) as f:
    cluster_color_name = json.load(f)

# =========================
# Extract clusters
# =========================

for cid in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == cid].copy()

    subset["cluster_color"] = cluster_color_name[str(cid)]
    subset["is_real"] = subset["Label"].isna()

    color_tag = cluster_color_name[str(cid)].replace(" ", "_")
    fname = f"cluster_{cid}_{color_tag}.csv"

    subset.to_csv(os.path.join(out_dir, fname), index=False)

    print(
        f"Cluster {cid} ({color_tag}): "
        f"{len(subset)} samples | "
        f"real={subset['is_real'].sum()} | "
        f"synthetic={(~subset['is_real']).sum()}"
    )

print("\nStep 3 complete (clusters extracted with color labels)")

