import pandas as pd
import numpy as np
import anndata

def process_crohn_dataset(file_path: str, output_name: str):
    print(f"📥 Loading {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Raw DataFrame shape (rows × cols): {df.shape}")

    # Drop metadata columns (ID_REF and IDENTIFIER)
    sample_columns = df.columns[2:]
    features = df[sample_columns]
    print(f"After dropping metadata → shape: {features.shape} (genes × samples)")

    # Convert all to numeric
    numeric_df = features.apply(pd.to_numeric, errors="coerce")
    print(f"After numeric conversion → shape: {numeric_df.shape}")

    # Transpose → samples × genes
    X = numeric_df.values.T
    print(f"Expression matrix shape (samples × genes): {X.shape}")

    # Handle NaNs
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        print(f"⚠️ Found {nan_count} NaN values → filling with zeros")
        X = np.nan_to_num(X, nan=0.0)

    # Normalize per gene
    print("📊 Normalizing data (mean=0, std=1 per gene)...")
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    print(f"Final normalized matrix shape: {X.shape}")

    # Create AnnData
    adata = anndata.AnnData(X)
    adata.obs_names = [f"sample_{i}" for i in range(X.shape[0])]
    adata.var_names = df["ID_REF"].astype(str) if "ID_REF" in df.columns else [f"gene_{i}" for i in range(X.shape[1])]
    adata.obs["celltype"] = np.zeros(adata.n_obs, dtype=int)

    out_path = f"data/{output_name}.h5ad"
    adata.write(out_path)
    print(f"✅ Saved to {out_path}")

# Example:
# process_crohn_dataset("/home/prathyusha/Hari/crohndataset.csv", "CrohnDataset")
process_crohn_dataset("data/crohndataset.csv", "crohnDataset")