# import anndata
# adata = anndata.read_h5ad("data/tabula_muris/all.h5ad")
# adata = anndata.read_h5ad("data/RNA-Seq/10X_PBMC.mat")
# print(adata.obs.columns)  # look for the label column name
# print(adata.obs["celltype"].unique())  # example label column
# dr = adata.obs["celltype"].unique()
# for k, i in enumerate(dr):
#     print(k, " ", i)
# print(type(adata))
# import pdb
# import scanpy as sc
# adata = sc.read_h5ad("data/10X_PBMC.h5ad")
# print(adata.shape)  # (cells, genes)
# pdb.set_trace()

# import scipy.io
# import anndata
# import numpy as np
# import h5py

# f = h5py.File("data/RNASeq/10X_PBMC.h5", "r")
# print(list(f.keys()))

# # Load data matrix and labels
# X = f['X'][:]
# Y = f['Y'][:]
# if Y.ndim > 1:
#     Y = Y[:, 0]  # Flatten if it's shaped like (n, 1)

# # Build AnnData object
# adata = anndata.AnnData(X)
# adata.obs['celltype'] = Y.astype(str)

# # Save as .h5ad
# adata.write("data/10X_PBMC.h5ad")
# print("✅ Saved to converted_data.h5ad")

# import pandas as pd
# import numpy as np
# import scanpy as sc
# import anndata

# # Load CSV
# df = pd.read_csv("data/RNASeq/BreastCancerDataset_.csv", index_col=0)
# df = df.T
# df = df.apply(pd.to_numeric, errors="coerce")
# df = df.dropna(axis=1)

# # Normalize and log1p
# X = df.values.astype(np.float32)
# X = X / np.clip(X.sum(axis=1, keepdims=True), 1e-6, None) * 1e4
# X = np.log1p(X)

# # Create AnnData
# adata = anndata.AnnData(X)
# adata.obs["fake_labels"] = ["cell_" + str(i) for i in range(X.shape[0])]
# adata.var_names = df.columns

# # Save to h5ad
# adata.write("data/BreastCancerDataset.h5ad")
# print("✅ Saved as: data/RNASeq/BreastCancerDataset_.h5ad")

# import pandas as pd

# df = pd.read_csv("data/RNASeq/BreastCancerDataset_.csv", index_col=0, low_memory=False)

# # Print column types
# print(df.dtypes.head(50))  # adjust the number as needed

# # Check how many unique types are in each column
# for col in df.columns[:50]:  # limit to first 50 for speed
#     types = df[col].apply(lambda x: type(x)).value_counts()
#     print(f"Column: {col}")
#     print(types)
#     print("-" * 30)

import pandas as pd
import numpy as np
import scanpy as sc
import anndata

# === Step 1: Load CSV ===
csv_path = "data/RNASeq/BreastCancerDataset_.csv"
print(f"📥 Loading CSV: {csv_path}")
df = pd.read_csv(csv_path, index_col=0)

# === Step 2: Transpose ===
df = df.T
print("🔄 Transposed. Shape now:", df.shape)

# === Step 3: Convert to numeric ===
df = df.apply(pd.to_numeric, errors="coerce")
print("🔢 Converted to numeric.")

# === Step 4: Drop non-numeric genes (columns) ===
before = df.shape[1]
df = df.dropna(axis=1)
after = df.shape[1]
print(f"🧹 Dropped {before - after} genes with non-numeric values.")
print("✅ Clean shape:", df.shape)

# === Step 5: Normalize & log1p ===
X = df.values.astype(np.float32)
# X = X / np.clip(X.sum(axis=1, keepdims=True), 1e-6, None) * 1e4
# X = np.log1p(X)

# === Step 6: Create AnnData and Save ===
adata = anndata.AnnData(X)
adata.obs_names = df.index
adata.var_names = df.columns

out_path = "data/BreastCancerDataset2.h5ad"
adata.write(out_path)
print(f"✅ Saved clean .h5ad to: {out_path}")


import scanpy as sc
adata = sc.read_h5ad("data/BreastCancerDataset2.h5ad")

print("✅ Columns in adata.obs:", adata.obs.columns)

import scanpy as sc
import numpy as np

# Load your .h5ad file
adata = sc.read_h5ad("data/BreastCancerDataset2.h5ad")

# Add dummy labels
adata.obs["celltype"] = np.zeros(adata.n_obs, dtype=int)

# Save it back
adata.write("data/BreastCancerDataset2.h5ad")

print("✅ 'celltype' column added with dummy labels")


