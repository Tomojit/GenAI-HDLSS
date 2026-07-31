import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance

# Set random seed for reproducibility
np.random.seed(42)

# ====== Load your data ======
# real = np.load("data/breast_cancer_svd_latent.npy")  # (15, 15)
# gen = np.load("output/simulated_samples/breastcancerSVDgenerated.npz")["cell_gen"]  # (200, 15)

# real = np.load("data/GLIOMA_svd_latent.npy")  # (15, 15)
# gen = np.load("output/simulated_samples/GLIOMASVDgenerated.npz")["cell_gen"]

real = np.load("data/colon_svd_latent.npy")  # (15, 15)
gen = np.load("output/simulated_samples/COLONSVDgenerated.npz")["cell_gen"]

name = 'RNA'

print(f"Real data shape: {real.shape}")
print(f"Generated data shape: {gen.shape}")

# ===========================
# 1. Define KL Divergence function with binning
# ===========================
def calculate_kl_divergence_binned(original_data, synthetic_data, bins=30):
    """
    Calculate KL divergence by binning the distance distributions.
    """
    centroid = np.mean(original_data, axis=0)
    original_distances = np.linalg.norm(original_data - centroid, axis=1)
    synthetic_distances = np.linalg.norm(synthetic_data - centroid, axis=1)

    # Create histograms
    Q_hist, bin_edges = np.histogram(original_distances, bins=bins, density=True)
    P_hist, _ = np.histogram(synthetic_distances, bins=bin_edges, density=True)

    # Small epsilon to avoid log(0)
    epsilon = 1e-10
    P_hist = np.clip(P_hist, epsilon, 1)
    Q_hist = np.clip(Q_hist, epsilon, 1)

    kl_value = np.sum(P_hist * np.log(P_hist / Q_hist))
    return kl_value

# ===========================
# 2. Calculate Divergences
# ===========================
# KL Divergence
kl_value = calculate_kl_divergence_binned(real, gen, bins=30)

# Wasserstein Distance
centroid_real = np.mean(real, axis=0)
dists_real = np.linalg.norm(real - centroid_real, axis=1)
dists_gen = np.linalg.norm(gen - centroid_real, axis=1)
wasserstein_value = wasserstein_distance(dists_real, dists_gen)

print(f"\nKL Divergence (binned): {kl_value:.4f}")
print(f"Wasserstein Distance: {wasserstein_value:.4f}\n")

# ===========================
# 3. Plot Distance Distributions
# ===========================
plt.figure(figsize=(8, 6))
sns.histplot(dists_real, color='blue', label='Real Data', kde=True, bins=15, alpha=0.6)
sns.histplot(dists_gen, color='red', label='Generated Data', kde=True, bins=15, alpha=0.6)
plt.title(f"Distance Distributions\nKL: {kl_value:.4f}, Wass: {wasserstein_value:.4f}")
plt.xlabel("Distance from Centroid")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("distance_distributions.png")
plt.show()