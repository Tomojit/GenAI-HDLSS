import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# ==========================================================
def SVD_RNA(loc):
    normalization_method = "zscore"  # options: "log1p", "zscore", "minmax"

    df = pd.read_csv(loc, low_memory=False)

    # 🔥 Drop useless columns
    if 'Sample_ID' in df.columns:
        df = df.drop('Sample_ID', axis=1)
    if 'Gender' in df.columns:
        df = df.drop('Gender', axis=1)

    # 🔥 Extract labels separately
    labels = df['Phenotype/Label'].to_numpy()

    # 🔥 Drop label column from features
    df = df.drop('Phenotype/Label', axis=1)

    # 🔥 Now df has only numeric features
    data_only = df.to_numpy().astype(float)

    # Mean subtraction (column-wise centering)
    data_only -= np.mean(data_only, axis=0)

    # Normalization
    if normalization_method == "log1p":
        row_sums = data_only.sum(axis=1, keepdims=True)
        normalized_data = data_only / (row_sums + 1e-8) * 1e4
        normalized_data = np.clip(normalized_data, 0, None)
        norm_data = np.log1p(normalized_data)
        print("✅ Used log1p normalization")

    elif normalization_method == "zscore":
        mean = np.mean(data_only, axis=0)
        std = np.std(data_only, axis=0)
        norm_data = (data_only - mean) / (std + 1e-8)
        print("✅ Used z-score normalization")

    elif normalization_method == "minmax":
        data_min = np.min(data_only, axis=0)
        data_max = np.max(data_only, axis=0)
        norm_data = (data_only - data_min) / (data_max - data_min + 1e-8)
        print("✅ Used min-max normalization")

    else:
        raise ValueError(f"Unknown normalization method: {normalization_method}")

    print("✅ Normalized data shape:", norm_data.shape)
    print("✅ Labels shape:", labels.shape)

    # Perform SVD
    U, S, V = np.linalg.svd(norm_data.T, full_matrices=False)
    return U, S, V, norm_data.T, labels

# ==========================================================
# Main script starts here

real = np.load("data/RNA_svd_latent.npy")  # optional
gen = np.load("output/simulated_samples/RNASVDgenerated.npz")["cell_gen"]
loc = "data/RNASeq/RNA.csv"

# Load real data correctly using RNA specific SVD
U, S, V, Dog, real_labels = SVD_RNA(loc)

# Project generated data
Dgen = np.matmul(U, gen.T)

# Transpose both to (samples, features)
Dog_T = Dog.T
Dgen_T = Dgen.T

print("Dog_T shape:", Dog_T.shape)
print("Dgen_T shape:", Dgen_T.shape)

# Combine real + generated
combined = np.vstack([Dog_T, Dgen_T])

# 2D PCA
pca_2d = PCA(n_components=2)
combined_2d = pca_2d.fit_transform(combined)

Dog_2d = combined_2d[:Dog_T.shape[0]]
Dgen_2d = combined_2d[Dog_T.shape[0]:]

# Plot 2D PCA
if not os.path.exists("pcaPlots"):
    os.makedirs("pcaPlots")

plt.figure(figsize=(8, 6))
plt.scatter(Dog_2d[:, 0], Dog_2d[:, 1], color='blue', alpha=0.6, label='Real')
plt.scatter(Dgen_2d[:, 0], Dgen_2d[:, 1], color='red', alpha=0.6, label='Generated')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("RNASVDgenerated - 2D PCA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pcaPlots/RNASVDgenerated2D.jpeg")
plt.close()

# 3D PCA
pca_3d = PCA(n_components=3)
combined_3d = pca_3d.fit_transform(combined)

Dog_3d = combined_3d[:Dog_T.shape[0]]
Dgen_3d = combined_3d[Dog_T.shape[0]:]

# Plot 3D PCA
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Dog_3d[:, 0], Dog_3d[:, 1], Dog_3d[:, 2], color='blue', alpha=0.6, label='Real')
ax.scatter(Dgen_3d[:, 0], Dgen_3d[:, 1], Dgen_3d[:, 2], color='red', alpha=0.6, label='Generated')
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.set_title("RNASVDgenerated - 3D PCA")
ax.legend()
plt.tight_layout()
plt.savefig("pcaPlots/RNASVDgenerated3D.jpeg")
plt.close()
