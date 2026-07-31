import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scanpy as sc
from VAE_model import VAE
import pdb

# ========== 1. Load generated latent samples ==========
npz_path = "output/simulated_samples/Crohn1LakhCKPT.npz"
# npz_path = "output/simulated_samples/breastCancer.npz"
data = np.load(npz_path)
latent_gen = data["cell_gen"]
print("Generated latent shape:", latent_gen.shape)

# ========== 2. Decode using trained VAE ==========
vae_ckpt = "output/checkpoint/AE/my_VAECrohn/model_seed=0_step=30000.pt"
# vae_ckpt = "output/checkpoint/AE/my_VAEBrCa/model_seed=0_step=19999.pt"
vae = VAE(num_genes=22284, device="cpu", hidden_dim=128)
# vae = VAE(num_genes=22283, device="cpu", hidden_dim=128)

vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
vae.eval()

with torch.no_grad():
    decoded_gen = vae(torch.tensor(latent_gen).float(), return_decoded=True).numpy()
print("Decoded generated shape:", decoded_gen.shape)

# ========== 3. Load real gene expression ==========
# adata = sc.read_h5ad("data/tabula_muris/all5000Strat.h5ad")
adata = sc.read_h5ad("data/crohnDataset.h5ad")

# adata = sc.read_h5ad("data/BreastCancerDataset.h5ad")

adata.var_names_make_unique()
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
import scipy.sparse as sp
# real_expr = adata.X.toarray()
X = adata.X
if sp.issparse(X):
    real_expr = X.toarray()
else:
    real_expr = np.asarray(X)
print("Real expression shape:", real_expr.shape)

# ========== 4. Subsample real data for comparison ==========
real_expr = real_expr[::1]   # take every 5th cell to reduce size
decoded_gen = decoded_gen[::1]  # use all generated
# np.savetxt("output.csv", decoded_gen, delimiter=",")
print(type(decoded_gen))
# pdb.set_trace()

# ========== DIAGNOSTICS: check for NaN/inf in each matrix ==========
def report_matrix(name, X, show_max=10):
    print(f"\n[DIAG] {name}: shape={X.shape}, dtype={X.dtype}")
    n_nan = np.isnan(X).sum()
    n_posinf = np.isposinf(X).sum() if np.isinf(X).any() else 0
    n_neginf = np.isneginf(X).sum() if np.isinf(X).any() else 0
    print(f"[DIAG] {name}: NaNs={n_nan}, +inf={n_posinf}, -inf={n_neginf}")
    if n_nan > 0:
        rows_nan = np.where(np.isnan(X).any(axis=1))[0]
        cols_nan = np.where(np.isnan(X).any(axis=0))[0]
        print(f"[DIAG] {name}: rows with NaN = {len(rows_nan)} (show up to {show_max}): {rows_nan[:show_max]}")
        print(f"[DIAG] {name}: cols with NaN = {len(cols_nan)} (show up to {show_max}): {cols_nan[:show_max]}")
    n_neg = (X < 0).sum()
    print(f"[DIAG] {name}: negative entries = {n_neg}")
    col_std = X.std(axis=0)
    zerovar_cols = np.where(col_std == 0)[0]
    if zerovar_cols.size:
        print(f"[DIAG] {name}: zero-variance cols = {zerovar_cols.size} (show up to {show_max}): {zerovar_cols[:show_max]}")

# run diagnostics on both sources
report_matrix("REAL (after scanpy normalize_total/log1p & toarray)", real_expr)
report_matrix("GENERATED (decoded_gen from VAE)", decoded_gen)

# ========== 5. Combine and run PCA ==========
combined = np.vstack([real_expr, decoded_gen])

# Extra check on combined
report_matrix("COMBINED (before PCA)", combined)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined)

# ========== 6. Plot ==========
n_real = real_expr.shape[0]
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:n_real, 0], pca_result[:n_real, 1], s=5, c='blue', label='Real')
plt.scatter(pca_result[n_real:, 0], pca_result[n_real:, 1], s=5, c='red', label='Generated')
plt.title("PCA: Crohn 127 Real vs 127 Generated Cells with 1Lakh CKPT")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.savefig("Crohn127Strat1LakhCKPT.jpeg", dpi=200, bbox_inches="tight")
plt.show()
np.savetxt("Crohn127Strat1LakhCKPT.csv", decoded_gen, delimiter=",")
plt.close()
