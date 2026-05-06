import pandas as pd
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import TruncatedSVD

# =========================
# STEP 1: LOAD DATA
# =========================
df = pd.read_excel("data.xlsx")

print("Original shape:", df.shape)

# =========================
# STEP 2: CLEAN DATA
# =========================
df = df.fillna("")

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

for col in df.columns:
    df[col] = df[col].apply(clean_text)

# Remove fully empty rows
df = df[
    (df["Patient Comments"] != "") |
    (df["Treatment Interest"] != "") |
    (df["Patient Information"] != "") |
    (df["Team Notes"] != "")
]

print("Cleaned shape:", df.shape)

# Save cleaned file
df.to_excel("cleaned_data.xlsx", index=False)
print("✅ cleaned_data.xlsx saved")

# =========================
# STEP 3: LOAD BGE-M3 MODEL
# =========================
model_name = "BAAI/bge-m3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Using device:", device)

# =========================
# STEP 4: EMBEDDING FUNCTION
# =========================
def get_embedding(text):
    if text.strip() == "":
        return np.zeros(1024)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.squeeze().cpu().numpy()

# =========================
# STEP 5: COLUMN EMBEDDINGS
# =========================
columns = [
    "Patient Comments",
    "Treatment Interest",
    "Patient Information",
    "Team Notes"
]

final_df = df.copy()

for col in columns:
    print(f"Embedding column: {col}")

    embeddings = df[col].apply(get_embedding)
    matrix = np.vstack(embeddings.values)

    emb_df = pd.DataFrame(
        matrix,
        columns=[f"{col}_emb_{i}" for i in range(matrix.shape[1])]
    )

    final_df = pd.concat([final_df, emb_df], axis=1)

# Save raw embeddings version
final_df.to_excel("embedded_data.xlsx", index=False)
print("✅ embedded_data.xlsx saved")

# =========================
# STEP 6: APPLY SVD
# =========================

# Extract only embedding columns
embedding_cols = [c for c in final_df.columns if "_emb_" in c]
X = final_df[embedding_cols].values

print("Original embedding shape:", X.shape)

# Reduce dimensions
svd = TruncatedSVD(n_components=200, random_state=42)
X_reduced = svd.fit_transform(X)

print("Reduced shape:", X_reduced.shape)

# Show information retained
print("Explained variance:", sum(svd.explained_variance_ratio_))

# =========================
# STEP 7: CREATE SVD DATAFRAME
# =========================
svd_df = pd.DataFrame(
    X_reduced,
    columns=[f"svd_{i}" for i in range(X_reduced.shape[1])]
)

final_svd_df = pd.concat(
    [final_df.reset_index(drop=True), svd_df],
    axis=1
)

# =========================
# STEP 8: SAVE FINAL SVD FILE
# =========================
final_svd_df.to_excel("svd_embedded_data.xlsx", index=False)

print("✅ svd_embedded_data.xlsx saved")
print("Final shape:", final_svd_df.shape) 