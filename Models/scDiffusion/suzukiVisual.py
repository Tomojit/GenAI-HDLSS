import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scanpy as sc
from VAE_model import VAE

# ========== CONFIGURATION ==========
CONFIG = {
    "npz_path": "output/simulated_samples/suzuki.npz",
    "vae_ckpt": "output/checkpoint/AE/suzuki_VAE/model_seed=0_step=10000.pt",
    "num_genes": 22206,
    "hidden_dim": 128,
    "h5ad_path": "data/suzuki.h5ad",
    "dataset_name": "SUZUKI",  # Used for plot title and output files
    "subsample_step": 1,  # Take every Nth cell (1 = no subsampling)
    "device": "cpu"
}
# ===================================

# ========== 1. Load generated latent samples ==========
data = np.load(CONFIG["npz_path"])
latent_gen = data["cell_gen"]
print("Generated latent shape:", latent_gen.shape)

# ========== 2. Decode using trained VAE ==========
vae = VAE(
    num_genes=CONFIG["num_genes"], 
    device=CONFIG["device"], 
    hidden_dim=CONFIG["hidden_dim"]
)
vae.load_state_dict(torch.load(CONFIG["vae_ckpt"], map_location=CONFIG["device"]))
vae.eval()

with torch.no_grad():
    decoded_gen = vae(torch.tensor(latent_gen).float(), return_decoded=True).numpy()
print("Decoded generated shape:", decoded_gen.shape)

# ========== 3. Load real gene expression ==========
adata = sc.read_h5ad(CONFIG["h5ad_path"])
adata.var_names_make_unique()

import scipy.sparse as sp
X = adata.X
if sp.issparse(X):
    real_expr = X.toarray()
else:
    real_expr = np.asarray(X)
print("Real expression shape:", real_expr.shape)

# ========== 4. Subsample real data for comparison ==========
real_expr = real_expr[::CONFIG["subsample_step"]]
decoded_gen = decoded_gen[::CONFIG["subsample_step"]]

# ========== DIAGNOSTICS ==========
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

report_matrix("REAL", real_expr)
report_matrix("GENERATED", decoded_gen)

# ========== 5. Combine and run PCA ==========
combined = np.vstack([real_expr, decoded_gen])
report_matrix("COMBINED", combined)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined)

# ========== 6. Plot ==========
n_real = real_expr.shape[0]
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:n_real, 0], pca_result[:n_real, 1], s=5, c='blue', label='Real')
plt.scatter(pca_result[n_real:, 0], pca_result[n_real:, 1], s=5, c='red', label='Generated')
plt.title(f"PCA: {CONFIG['dataset_name']} Real vs Generated Cells")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.savefig(f"{CONFIG['dataset_name']}_comparison.jpeg", dpi=200, bbox_inches="tight")
plt.show()
np.savetxt(f"{CONFIG['dataset_name']}_output.csv", decoded_gen, delimiter=",")
plt.close()