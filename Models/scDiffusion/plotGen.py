
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scanpy as sc
from VAE_model import VAE
import pdb

# ========== 1. Load generated latent samples ==========
# npz_path = "output/simulated_samples/Crohn1LakhCKPT.npz"
# npz_path = "output/simulated_samples/RNA7LakhCKPTNoLog.npz"
# npz_path = "output/simulated_samples/breastCancer7LakhCKPTNoLog.npz"
# npz_path = "output/simulated_samples/Crohn6LakhCKPTNoLog.npz"
# npz_path = "output/simulated_samples/muris1000Strat8LakhCKPT.npz"
# npz_path = "output/simulated_samples/muris50002LakhCKPT.npz"
npz_path = "output/simulated_samples/muris100002LakhCKPT.npz"
# npz_path = "output/simulated_samples/muris50Strat.npz"


data = np.load(npz_path)
latent_gen = data["cell_gen"]
print("Generated latent shape:", latent_gen.shape)

# ========== 2. Decode using trained VAE ==========
# vae_ckpt = "output/checkpoint/AE/my_VAECrohn/model_seed=0_step=30000.pt"
# vae = VAE(num_genes=22284, device="cpu", hidden_dim=128)

# vae_ckpt = "output/checkpoint/AE/my_VAEBrCa/model_seed=0_step=30000.pt"
# vae = VAE(num_genes=22284, device="cpu", hidden_dim=128)

# vae_ckpt = "output/checkpoint/AE/my_VAERNA/model_seed=0_step=30000.pt"
# vae = VAE(num_genes=6658, device="cpu", hidden_dim=128)
# vae = VAE(num_genes=22283, device="cpu", hidden_dim=128)

# vae_ckpt = "output/checkpoint/AE/my_VAE1000Strat/model_seed=0_step=199999.pt"
# vae = VAE(num_genes=18996, device="cpu", hidden_dim=128)

# vae_ckpt = "output/checkpoint/AE/my_VAE5000Strat/model_seed=0_step=199999.pt"
# vae = VAE(num_genes=18996, device="cpu", hidden_dim=128)

vae_ckpt = "output/checkpoint/AE/my_VAE10000Strat/model_seed=0_step=200000.pt"
vae = VAE(num_genes=18996, device="cpu", hidden_dim=128)

# vae_ckpt = "output/checkpoint/AE/my_VAE50Strat/model_seed=0_step=19999.pt"
# vae = VAE(num_genes=18996, device="cpu", hidden_dim=128)

vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"))
vae.eval()

with torch.no_grad():
    decoded_gen = vae(torch.tensor(latent_gen).float(), return_decoded=True).numpy()
print("Decoded generated shape:", decoded_gen.shape)

# ========== 3. Load real gene expression ==========
# adata = sc.read_h5ad("data/tabula_muris/all5000Strat.h5ad")
adata = sc.read_h5ad("data/tabula_muris/all10000Strat.h5ad")
# adata = sc.read_h5ad("data/tabula_muris/all50Strat.h5ad")
# adata = sc.read_h5ad("data/tabula_muris/all1000Strat.h5ad")

# adata = sc.read_h5ad("data/RNADataset.h5ad")

# adata = sc.read_h5ad("data/BreastCancerDataset2.h5ad")

# adata = sc.read_h5ad("data/crohnDataset.h5ad")


adata.var_names_make_unique()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
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
# ========== 5. Combine and run PCA ==========
combined = np.vstack([real_expr, decoded_gen])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined)
# np.savetxt("pcaCrohn.csv", pca_result, delimiter=",")
# import pdb
# pdb.set_trace()

# ========== 6. Plot ==========
n_real = real_expr.shape[0]
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:n_real, 0], pca_result[:n_real, 1], s=5, c='blue', label='Real')
plt.scatter(pca_result[n_real:, 0], pca_result[n_real:, 1], s=5, c='red', label='Generated')
# plt.title("PCA: Crohn 127 Real vs 127 Generated Cells with 1Lakh CKPT")
# plt.title("PCA: RNA 15 Real vs 15 Generated Cells with 1Lakh CKPT")

# plt.title("PCA: RNA 15 Real vs 15 Generated Cells with 7Lakh CKPT No Log")

# plt.title("PCA: BC 42 Real vs 42 Generated Cells with 7Lakh CKPT No Log")

# plt.title("PCA: muris 50 Real vs 50 Generated Cells with 20Thousand CKPT")
# plt.title("PCA: muris 1000 Real vs 1000 Generated Cells with 2Lakh CKPT")
# plt.title("PCA: muris 5000 Real vs 5000 Generated Cells with 2Lakh CKPT")
plt.title("PCA: muris 10000 Real vs 10000 Generated Cells with 2Lakh CKPT")


plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("RNA15Strat1LakhCKPT.jpeg")
# np.savetxt("RNAStrat1LakhCKPT.csv", decoded_gen, delimiter=",")
# np.savetxt("pcaCrohn.csv", pca_result, delimiter=",")


# plt.savefig("RNA15Strat7LakhCKPTNoLog.jpeg")
# np.savetxt("RNAStrat7LakhCKPTNoLog.csv", decoded_gen, delimiter=",")
# np.savetxt("pcaRNA.csv", pca_result, delimiter=",")

# plt.savefig("BC42Strat7LakhCKPTNoLog.jpeg")
# np.savetxt("BCS42trat7LakhCKPTNoLog.csv", decoded_gen, delimiter=",")
# np.savetxt("pcaBC.csv", pca_result, delimiter=",")

# plt.savefig("Crohn127Strat6LakhCKPTNoLog.jpeg")
# np.savetxt("Crohn127trat6LakhCKPTNoLog.csv", decoded_gen, delimiter=",")
# np.savetxt("pcaCrohn.csv", pca_result, delimiter=",")

# plt.savefig("Muris5000Strat2LakhCKPT15Sept.jpeg")
# np.savetxt("Muris5000Strat2LakhCKPT15Sept.csv", decoded_gen, delimiter=",")
# np.savetxt("pcaMuris5000.csv", pca_result, delimiter=",")


plt.savefig("Muris10000Strat2LakhCKPT15Sept.jpeg")
np.savetxt("Muris10000Strat2LakhCKPT15Sept.csv", decoded_gen, delimiter=",")
np.savetxt("pcaMuris10000.csv", pca_result, delimiter=",")

# plt.savefig("Muris1000Strat2LakhCKPT15Sept.jpeg")
# np.savetxt("Muris1000Strat2LakhCKPT15Sept.csv", decoded_gen, delimiter=",")
# np.savetxt("pcaMuris1000.csv", pca_result, delimiter=",")

# plt.savefig("Muris50Strat20ThousandCKPT15Sept.jpeg")
# np.savetxt("Muris50Strat20ThousandCKPT15Sept.csv", decoded_gen, delimiter=",")
# np.savetxt("pcaMuris50.csv", pca_result, delimiter=",")

plt.close()