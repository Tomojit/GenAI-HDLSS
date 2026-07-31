# convert_suzuki.py
import pandas as pd
import numpy as np
import anndata

# Load raw data
df = pd.read_csv('/path/to/suzuki.csv')
print("Raw shape:", df.shape)

# Drop ID_REF column and Label column, transpose to cells x genes
gene_names = df.iloc[0, 1:-1].values  # gene names from first row, skip ID_REF and Label
df = df.T
df = df.iloc[1:, :-1]  # drop header row and label column
print("After transpose and trim:", df.shape)

# Convert to numeric
X = df.values.astype("float64")

# Z-score normalize per gene
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
print(f"Normalized shape: {X.shape}, min: {X.min():.3f}, max: {X.max():.3f}")

# Create AnnData
adata = anndata.AnnData(X=X.astype("float32"))
adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
adata.var_names = [str(g) for g in gene_names]
adata.obs["celltype"] = 0

adata.write("datah5ad/suzuki.h5ad")
print("Saved:", adata.shape)