import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================
# Config
# =========================

FILE = "/Users/aysasorahi/Documents/master/SLAM LAB/REZA/data/LOOK_data.ods"
OUT_DIR = "intra_function_artifacts"
TARGET_FUNC = "INTJ"        # change to DIR or DM
AUG_LABEL = "L"             # augmented samples
MODEL_NAME = "bert-base-uncased"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Load model
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# =========================
# Load data (ODS)
# =========================

df = pd.read_excel(FILE, engine="odf")

# Filter: function + augmented only
df = df[
    (df["function"] == TARGET_FUNC) &
    (df["Label"] == AUG_LABEL)
].reset_index(drop=True)

print(
    f"Loaded {len(df)} augmented samples "
    f"for function = {TARGET_FUNC}"
)

# =========================
# Extract embeddings
# =========================

embeddings = []
kept_rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    sentence = row["sample"]

    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**encoded)

    hidden_states = outputs.last_hidden_state.squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(
        encoded["input_ids"].squeeze(0)
    )

    # Find token 'look'
    if "look" not in tokens:
        continue

    look_idx = tokens.index("look")
    look_embedding = hidden_states[look_idx].numpy()

    embeddings.append(look_embedding)
    kept_rows.append(row)

# =========================
# Save outputs
# =========================

embeddings = np.vstack(embeddings)

emb_path = f"{OUT_DIR}/{TARGET_FUNC}_L_embeddings.npy"
np.save(emb_path, embeddings)

out_df = pd.DataFrame(kept_rows)
csv_path = f"{OUT_DIR}/{TARGET_FUNC}_L_texts.csv"
out_df.to_csv(csv_path, index=False)

print(f"\nSaved embeddings to: {emb_path}")
print(f"Saved texts to: {csv_path}")
print("\n Embedding extraction complete.")

