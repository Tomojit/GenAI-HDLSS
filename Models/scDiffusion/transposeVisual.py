import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === Load original data ===
original = np.load("data/RNA_svd_latent.npy")  # shape: (6658, 15)
print("✅ Loaded original data:", original.shape)

# === Load generated data ===
gen_npz = np.load("output/simulated_samples/RNASVDgenerated.npz")

# Check available keys
print("Available keys in .npz:", gen_npz.files)

# You might see something like: ['samples'] or ['my_generated'] — replace below if needed
generated = gen_npz[gen_npz.files[0]]  # Automatically get the first key
print("✅ Loaded generated data:", generated.shape)

# === Combine both for joint PCA ===
original = original.T 
generated = generated.T
combined = np.concatenate([original, generated], axis=0)

# === Perform PCA ===
pca = PCA(n_components=2)
combined_2d = pca.fit_transform(combined)

# Split them back
orig_2d = combined_2d[:original.shape[0]]
gen_2d = combined_2d[original.shape[0]:]

# === Plot ===
plt.figure(figsize=(8, 6))
plt.scatter(orig_2d[:, 0], orig_2d[:, 1], s=5, c='blue', label="Original", alpha=0.5)
plt.scatter(gen_2d[:, 0], gen_2d[:, 1], s=10, c='red', label="Generated", alpha=0.8, marker='x')
plt.title("PCA Projection (2D) of Original (Blue) vs Generated (Red)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("RNAtransposeGenerated.jpeg")