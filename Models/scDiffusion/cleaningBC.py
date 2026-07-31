import pandas as pd
import numpy as np
import anndata

df = pd.read_csv('/home/ghoshstudents/VijayNewbase/tghosh-vijay-ConvexLDA/scDiffusion/datah5ad/BreastCancerDataset.csv')
print("Raw shape:", df.shape)

df = df.T
df = df.iloc[1:, :-1]
print("After transpose and trim:", df.shape)

X = df.values.astype("float64")

# z-score per gene
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
print(f"Normalized shape: {X.shape}, min: {X.min():.3f}, max: {X.max():.3f}")

adata = anndata.AnnData(X=X.astype("float32"))
adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
adata.var_names = [f"gene_{i}" for i in range(X.shape[1])]
adata.obs["celltype"] = 0

adata.write("datah5ad/breastCancer.h5ad")
print("Saved:", adata.shape)