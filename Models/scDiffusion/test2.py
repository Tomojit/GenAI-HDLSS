import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scanpy as sc
from VAE_model import VAE

# Load generated latent samples
data = np.load("output/simulated_samples/breastcancerGenerated.npz")
latent_gen = data["cell_gen"]
print("Generated latent shape:", latent_gen.shape)

# Load trained VAE and decode
vae_ckpt = "output/vae_checkpoints/BREASTCANCER/model_seed=0_step=199999.pt"
vae = VAE(num_genes=22283, device="cpu", hidden_dim=128)
vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
vae.eval()

with torch.no_grad():
    decoded_gen = vae(torch.tensor(latent_gen).float(), return_decoded=True).numpy()

# Load real expression
adata = sc.read_h5ad("data/BreastCancerDataset.h5ad")
adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

real_expr = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
print("✅ Real expression shape:", real_expr.shape)

# ✅ PCA only on real data
pca = PCA(n_components=2)
pca.fit(real_expr)

real_pca = pca.transform(real_expr)
gen_pca = pca.transform(decoded_gen)

# ✅ Plot both
plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], s=10, c='blue', label='Real')
plt.scatter(gen_pca[:, 0], gen_pca[:, 1], s=10, c='red', label='Generated')
plt.title("PCA: Real vs Generated Cells (Same Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("breast_real_vs_generated_FIXED.jpeg")
print("✅ Saved PCA plot as breast_real_vs_generated_FIXED.jpeg")

# ✅ Plot real only
plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], s=10, c='blue')
plt.title("PCA: Real Cells Only (Same Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("breast_real_only_FIXED.jpeg")
print("✅ Saved PCA plot as breast_real_only_FIXED.jpeg")
