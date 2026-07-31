import numpy as np
import scanpy as sc

full = sc.read_h5ad("data/tabula_muris/all.h5ad")
full.var_names_make_unique()

# stratified 50 indices by celltype (no preprocessing)
from sklearn.model_selection import StratifiedShuffleSplit
celltypes = full.obs["celltype"].values
sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
(train_idx, _), = sss.split(np.zeros(full.n_obs), celltypes)

# **CRITICAL**: subset rows (cells) ONLY; keep all genes (columns)
adata_50 = full[train_idx, :].copy()
print("Expect (50, 18996):", adata_50.shape)  # should be (50, 18996)

adata_50.write("./data/tabula_muris/all10000Strat.h5ad")

# ad = sc.read_h5ad("./data/tabula_muris/all.h5ad")
# ad = sc.read_h5ad("./data/BreastCancerDataset.h5ad")
ad = sc.read_h5ad("./data/tabula_muris/all10000Strat.h5ad")
# ad = sc.read_h5ad("./data/RNADataset.h5ad")

print("BreastCancerDataset.h5ad shape:", ad.shape)
