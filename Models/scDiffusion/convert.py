# fix_convert.py
# import pandas as pd
# import anndata
# import numpy as np
# import os

# os.makedirs("datah5ad", exist_ok=True)

# def convert(csv_path, out_path, dataset_name):
#     df = pd.read_csv(csv_path)  # no index_col
#     print(f"{dataset_name}: {df.shape[0]} cells x {df.shape[1]} genes")
    
#     adata = anndata.AnnData(X=df.values.astype("float32"))
#     adata.obs_names = [f"cell_{i}" for i in range(df.shape[0])]
#     adata.var_names = [str(g) for g in df.columns]
#     adata.obs["celltype"] = 0
    
#     adata.write(out_path)
#     print(f"Saved to {out_path}")

# convert("basicDiff/datasets/crohn.csv", "datah5ad/crohn.h5ad", "Crohn")
# convert("basicDiff/datasets/breastCancer.csv", "datah5ad/breastCancer.h5ad", "BreastCancer")

import pandas as pd
import numpy as np
import anndata

df = pd.read_csv('/home/ghoshstudents/VijayNewbase/tghosh-vijay-ConvexLDA/scDiffusion/datah5ad/BreastCancerDataset.csv')
df= df.T
df = df.iloc[1:, :-1]
print(df.iloc[0,0])
X = df.values.astype("float64")

# z-score normalize per gene (same as Crohn)
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
print(f"After normalization - min: {X.min():.3f}, max: {X.max():.3f}, mean: {X.mean():.3f}")

adata = anndata.AnnData(X=X.astype("float32"))
adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
adata.var_names = [str(g) for g in df.columns]
adata.obs["celltype"] = 0

adata.write("datah5ad/breastCancer.h5ad")
print("Saved datah5ad/breastCancer.h5ad")