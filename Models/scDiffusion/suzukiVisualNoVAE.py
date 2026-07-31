import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scanpy as sc
import scipy.sparse as sp

# ========== CONFIGURATION ==========
# CONFIG = {
#     "npz_path": "output/simulated_samples/suzukiTranspose.npz",
#     "h5ad_path": "data/suzukiTranspose.h5ad",
#     "dataset_name": "SUZUKI_Transpose 512 512 256 128 Architecture",
#     "subsample_step": 1
# }

CONFIG = {
    "npz_path": "output/simulated_samples/suzukiSVD.npz",
    "h5ad_path": "data/suzukiSVD.h5ad",
    "dataset_name": "SUZUKI Timestep 500 50 samples",
    "subsample_step": 1
}
# ===================================

# ========== 1. Load generated gene expression ==========
data = np.load(CONFIG["npz_path"])
generated_expr = data["cell_gen"]  # This is already gene expression, not latent!
print("Generated expression shape:", generated_expr.shape)

# ========== 2. Load real gene expression ==========
adata = sc.read_h5ad(CONFIG["h5ad_path"])
adata.var_names_make_unique()

X = adata.X
if sp.issparse(X):
    real_expr = X.toarray()
else:
    real_expr = np.asarray(X)
print("Real expression shape:", real_expr.shape)

# ========== 3. Subsample for comparison ==========
real_expr = real_expr[::CONFIG["subsample_step"]]
generated_expr = generated_expr[::CONFIG["subsample_step"]]

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
report_matrix("GENERATED", generated_expr)

# ========== 4. Combine and run PCA ==========
combined = np.vstack([real_expr, generated_expr])
report_matrix("COMBINED", combined)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined)

# ========== 5. Plot ==========
n_real = real_expr.shape[0]
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:n_real, 0], pca_result[:n_real, 1], s=5, c='blue', label='Real', alpha=0.6)
plt.scatter(pca_result[n_real:, 0], pca_result[n_real:, 1], s=5, c='red', label='Generated', alpha=0.6)
plt.title(f"PCA: {CONFIG['dataset_name']} Real vs Generated Cells")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.savefig(f"{CONFIG['dataset_name']}_comparison.jpeg", dpi=200, bbox_inches="tight")
plt.show()
np.savetxt(f"{CONFIG['dataset_name']}_output.csv", generated_expr, delimiter=",")
plt.close()