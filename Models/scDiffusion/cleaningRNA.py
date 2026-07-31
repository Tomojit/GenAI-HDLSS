import pandas as pd
import numpy as np
import anndata
import scanpy as sc

def process_rna_dataset(file_path: str, output_name: str):
    print(f"📥 Loading {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Shape: {df.shape}")

    # 1. Identify feature columns
    if any(col.startswith("ID_REF_") for col in df.columns):
        feature_cols = [col for col in df.columns if col.startswith("ID_REF_")]
        features_df = df[feature_cols]
        print(f"🧬 Using {len(feature_cols)} ID_REF_ columns as features")
    else:
        # fallback: assume everything except first column is numeric features
        feature_cols = df.columns[1:]
        features_df = df[feature_cols]
        print(f"⚠️ No ID_REF_ columns found. Using {len(feature_cols)} columns from index 1:")

    # 2. Convert to numeric and impute NaNs with column mean
    numeric_df = features_df.apply(pd.to_numeric, errors="coerce")
    nan_count = numeric_df.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️ Found {nan_count} NaN values → imputing with column mean")
        numeric_df = numeric_df.fillna(numeric_df.mean())

    # 3. Expression matrix
    X = numeric_df.values.astype(np.float32)
    print(f"Raw expression matrix shape (samples × features): {X.shape}")

    # 4. Normalize (standardization per gene/column)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / (X_std + 1e-8)
    print("📊 Standardization complete (mean=0, std=1 per feature)")

    # 5. Set identifiers
    sample_ids = df["Sample_ID"].astype(str) if "Sample_ID" in df.columns else df.index.astype(str)
    gene_ids = feature_cols

    # 6. Create AnnData and save
    adata = anndata.AnnData(X)
    adata.obs_names = sample_ids
    adata.var_names = gene_ids
    adata.obs["celltype"] = np.zeros(adata.n_obs, dtype=int)

    out_path = f"data/{output_name}.h5ad"
    adata.write(out_path)
    print(f"✅ Saved to {out_path}")

# ========================
# Example usage:
# ========================
# process_rna_dataset("data/RNASeq/BreastCancerDataset_.csv", "BreastCancerDataset2")
process_rna_dataset("data/RNA_Dataset.csv", "RNADataset")
