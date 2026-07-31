import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import entropy, wasserstein_distance

# === Load data ===
real = np.load("data/breast_cancer_svd_latent.npy")
gen = np.load("output/simulated_samples/breastcancerSVDgenerated.npz")["cell_gen"]

print("✅ Real shape:", real.shape)
print("✅ Generated shape:", gen.shape)

# === PCA for 2D plot ===
pca2 = PCA(n_components=2)
pca2.fit(real)
real_2d = pca2.transform(real)
gen_2d = pca2.transform(gen)

# === Plot real vs generated ===
plt.figure(figsize=(8, 6))
plt.scatter(real_2d[:, 0], real_2d[:, 1], s=30, c='blue', label='Real')
plt.scatter(gen_2d[:, 0], gen_2d[:, 1], s=30, c='red', label='Generated')
plt.title("PCA (2D): Real vs Generated (SVD Latents)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_svd_real_vs_generated.jpeg")
print("✅ Saved: pca_svd_real_vs_generated.jpeg")

# === Plot real only ===
plt.figure(figsize=(8, 6))
plt.scatter(real_2d[:, 0], real_2d[:, 1], s=30, c='blue')
plt.title("PCA (2D): Real Only")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_svd_real_only.jpeg")
print("✅ Saved: pca_svd_real_only.jpeg")

# === PCA for metrics (10D) ===
pca10 = PCA(n_components=10)
real_10d = pca10.fit_transform(real)
gen_10d = pca10.transform(gen)

# === KL and Wasserstein per PC ===
kl_divs = []
wass_dists = []

for i in range(10):
    rh, rb = np.histogram(real_10d[:, i], bins=50, density=True)
    gh, _ = np.histogram(gen_10d[:, i], bins=rb, density=True)
    rh += 1e-10
    gh += 1e-10
    kl_divs.append(entropy(rh, gh))
    wass_dists.append(wasserstein_distance(real_10d[:, i], gen_10d[:, i]))

# === ELBO-like recon error (via PCA 10D)
real_recon = pca10.inverse_transform(real_10d)
gen_recon = pca10.inverse_transform(gen_10d)
real_err = np.mean((real - real_recon) ** 2)
gen_err = np.mean((gen - gen_recon) ** 2)

# === Print scores ===
print("\n📊 PCA Similarity Metrics:")
print("PCA\tKL Divergence\tWasserstein Distance")
for i in range(10):
    print(f"{i+1:02}\t{kl_divs[i]:.4f}\t\t{wass_dists[i]:.4f}")

print("\n🧠 PCA Reconstruction Error (MSE, ELBO proxy):")
print(f"Real: {real_err:.4f}")
print(f"Gen : {gen_err:.4f}")
