import anndata
adata = anndata.read_h5ad("data/tabula_muris/muris_50_subset.h5ad")
print("adata.shape:", adata.shape)

# Now check actual number of features
if hasattr(adata.X, "shape"):
    print("adata.X shape:", adata.X.shape)
else:
    print("adata.X is not matrix?")
