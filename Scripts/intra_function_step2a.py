import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import matplotlib.cm as cm
import os
import json

# =========================
# Config
# =========================

TARGET_FUNC = "DIR"      # change to DIR or DM
OUT_DIR = "intra_function_artifacts"

# =========================
# Load data
# =========================

emb = np.load(f"{OUT_DIR}/{TARGET_FUNC}_embeddings.npy")
df = pd.read_csv(f"{OUT_DIR}/{TARGET_FUNC}_hdbscan_clusters.csv")

print(f"Loaded {len(df)} samples for {TARGET_FUNC}")

# =========================
# UMAP
# =========================

reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.15,
    metric="euclidean",
    random_state=42
)

coords = reducer.fit_transform(emb)
df["umap_x"] = coords[:, 0]
df["umap_y"] = coords[:, 1]

# =========================
# Cluster → color mapping (EXPLICIT)
# =========================

cluster_ids = sorted(df["cluster"].unique())
cluster_ids_no_noise = [c for c in cluster_ids if c != -1]

cmap = cm.get_cmap("tab10", len(cluster_ids_no_noise))

cluster_color_map = {}
cluster_color_name = {}

for i, cid in enumerate(cluster_ids_no_noise):
    rgba = cmap(i)
    cluster_color_map[cid] = rgba
    cluster_color_name[cid] = f"tab10_color_{i}"

cluster_color_map[-1] = (0.6, 0.6, 0.6, 0.6)
cluster_color_name[-1] = "gray (noise)"

# Save mapping for Step 3
cluster_color_name_str = {str(k): v for k, v in cluster_color_name.items()}

with open(f"{OUT_DIR}/{TARGET_FUNC}_cluster_color_map.json", "w") as f:
    json.dump(cluster_color_name_str, f, indent=2)


# =========================
# Plot
# =========================

plt.figure(figsize=(10, 8))

# Synthetic
syn = df[df["Label"] == "L"]
plt.scatter(
    syn["umap_x"], syn["umap_y"],
    c=[cluster_color_map[c] for c in syn["cluster"]],
    s=35, alpha=0.5, marker="o", zorder=1
)

# Real
real = df[df["Label"].isna()]
plt.scatter(
    real["umap_x"], real["umap_y"],
    facecolors="white",
    edgecolors="black",
    linewidths=2.5,
    s=260,
    marker="^",
    zorder=10
)

# -------------------------
# Legend: cluster colors
# -------------------------

legend_elements = []
for cid in cluster_ids:
    legend_elements.append(
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=f"Cluster {cid} ({cluster_color_name[cid]})",
            markerfacecolor=cluster_color_map[cid],
            markersize=10
        )
    )

plt.legend(
    handles=legend_elements,
    title="HDBSCAN clusters",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.title(f"{TARGET_FUNC} — UMAP with explicit cluster colors")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()

out_fig = f"{OUT_DIR}/{TARGET_FUNC}_umap_with_cluster_legend.png"
plt.savefig(out_fig, dpi=300)
plt.show()

print(" Step 2 complete (with explicit cluster–color mapping)")

