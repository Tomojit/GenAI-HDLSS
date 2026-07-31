import pandas as pd
import numpy as np
import scanpy as sc
import anndata
from typing import Tuple

def _clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame by removing non-numeric characters commonly found in CSVs,
    then coercing to numeric.
    """
    # Strip whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Remove thousands separators and any stray spaces
    df = df.replace({r",": "", r"\s+": ""}, regex=True)
    # Coerce everything to numeric where possible
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def _choose_orientation(df_raw: pd.DataFrame, min_non_nan_ratio: float = 0.8) -> Tuple[pd.DataFrame, str]:
    """
    Try both orientations (as-is and transposed). Choose the one with more
    columns that have at least `min_non_nan_ratio` numeric (non-NaN) values.
    """
    def score(df):
        col_non_nan = df.notna().sum(axis=0)
        return (col_non_nan / len(df) >= min_non_nan_ratio).sum()

    # Clean both orientations separately (so coercion is consistent)
    df_a = _clean_numeric_frame(df_raw.copy())
    df_b = _clean_numeric_frame(df_raw.T.copy())

    score_a = score(df_a)
    score_b = score(df_b)

    if score_b > score_a:
        return df_b, "transposed"
    else:
        return df_a, "original"

def process_dataset(csv_path: str, output_name: str):
    """
    Process any RNA-seq dataset from CSV to h5ad

    Steps:
      - Load CSV as strings (safer for cleaning)
      - Auto-pick orientation that yields more numeric gene columns
      - Drop columns that are all-NaN; fill remaining NaN with 0
      - Print min/max before and after standardization
      - Save .h5ad and add dummy 'celltype'
    """
    print(f"📥 Loading CSV: {csv_path}")
    # Read everything as string to avoid partial dtype inference issues
    df_raw = pd.read_csv(csv_path, index_col=0, low_memory=False, dtype=str)
    print("📐 Raw CSV shape (rows, cols):", df_raw.shape)

    # Decide orientation automatically
    df, picked = _choose_orientation(df_raw, min_non_nan_ratio=0.8)
    print(f"↔️  Orientation chosen: {picked}. Shape now: {df.shape}")

    # Drop columns that are completely NaN (keep partially-valid columns)
    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    after_cols = df.shape[1]
    print(f"🧹 Dropped {before_cols - after_cols} all-NaN columns. Current shape: {df.shape}")

    # If everything is gone, bail out gracefully with a hint
    if df.shape[1] == 0 or df.shape[0] == 0:
        print("❌ No usable numeric data found after cleaning. "
              "Tips: check if the file needs transposing, remove headers/footers, or ensure numbers aren't mixed with text.")
        return

    # Fill remaining NaNs with 0 (sensible for sparse RNA-seq matrices)
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        print(f"🩹 Filling {nan_count} NaNs with 0.")
        df = df.fillna(0)

    # === Stats BEFORE standardization ===
    print("📏 Final dataset shape (cells, genes):", df.shape)
    try:
        before_min = float(np.nanmin(df.values))
        before_max = float(np.nanmax(df.values))
        print("🔽 Min value (before std):", before_min)
        print("🔼 Max value (before std):", before_max)
    except ValueError:
        print("⚠️ Could not compute min/max before std (empty or all-NaN).")

    # === Standardization: per gene (column) ===
    X = df.values.astype(np.float32)
    col_mean = np.nanmean(X, axis=0)
    col_std = np.nanstd(X, axis=0)
    # Avoid division by zero (genes with constant value)
    col_std_safe = np.where(col_std == 0, 1.0, col_std)
    X_std = (X - col_mean) / col_std_safe
    print("📊 Applied standardization (per gene: subtract mean, divide by std with 0-std guard)")

    # === Stats AFTER standardization ===
    try:
        after_min = float(np.min(X_std))
        after_max = float(np.max(X_std))
        print("🔽 Min value (after std):", after_min)
        print("🔼 Max value (after std):", after_max)
    except ValueError:
        print("⚠️ Could not compute min/max after std.")

    # === Create AnnData and Save ===
    adata = anndata.AnnData(X_std)
    adata.obs_names = df.index.astype(str)
    adata.var_names = df.columns.astype(str)

    out_path = f"data/{output_name}.h5ad"
    adata.write(out_path)
    print(f"✅ Saved clean .h5ad to: {out_path}")

    # === Add dummy labels ===
    adata = sc.read_h5ad(out_path)
    adata.obs["celltype"] = np.zeros(adata.n_obs, dtype=int)
    adata.write(out_path)
    print("✅ 'celltype' column added with dummy labels")

# === Usage ===
# Example: 
process_dataset("data/RNASeq/BreastCancerDataset_.csv", "BreastCancerDataset2")
# Your case:
# process_dataset("data/RNA_Dataset.csv", "RNADataset")
