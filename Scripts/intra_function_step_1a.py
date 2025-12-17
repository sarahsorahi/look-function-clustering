import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
import os

# =========================
# Config
# =========================

TARGET_FUNC = "DIR"      # change to DIR or DM
OUT_DIR = "intra_function_artifacts"

# =========================
# Load embeddings + metadata
# =========================

emb_path = f"{OUT_DIR}/{TARGET_FUNC}_embeddings.npy"
text_path = f"{OUT_DIR}/{TARGET_FUNC}_texts.csv"

if not os.path.exists(emb_path):
    raise FileNotFoundError(f"Missing embeddings: {emb_path}")

if not os.path.exists(text_path):
    raise FileNotFoundError(f"Missing metadata: {text_path}")

emb = np.load(emb_path)
df = pd.read_csv(text_path)

print(f"Loaded {len(df)} samples for {TARGET_FUNC}")
print("Augmented (Label == 'L'):", (df["Label"] == "L").sum())
print("Real (Label empty):", df["Label"].isna().sum())

# =========================
# HDBSCAN clustering
# =========================

print("\nRunning HDBSCAN...")

hdb = HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    metric="euclidean"
)

cluster_labels = hdb.fit_predict(emb)
df["cluster"] = cluster_labels

# =========================
# Cluster statistics
# =========================

unique_clusters = sorted(set(cluster_labels))
n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
n_noise = (cluster_labels == -1).sum()

print("\nClustering results:")
print("Clusters found:", n_clusters)
print("Noise points:", n_noise)

print("\nReal vs Synthetic per cluster:")
print(
    df.groupby("cluster")["Label"]
      .apply(lambda x: pd.Series({
          "synthetic_L": (x == "L").sum(),
          "real": x.isna().sum()
      }))
      .sort_index()
)

# =========================
# Save outputs
# =========================

out_csv = f"{OUT_DIR}/{TARGET_FUNC}_hdbscan_clusters.csv"
df.to_csv(out_csv, index=False)

print(f"\nSaved clustered data to: {out_csv}")
print(" Step 1 complete")

