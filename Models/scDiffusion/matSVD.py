import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy.io as sio

# === CONFIGURATION ===
normalization_method = "zscore"  # options: "log1p", "zscore", "minmax"
data_path = "data/colon.mat"
save_dir = "data/"
# ======================

# 1. Load Dataset
D = sio.loadmat(data_path)
data_only = D['X'].astype(float)  # Important: convert to float!
labels = D['Y']

# 2. Column-wise Mean Centering
data_only -= np.mean(data_only, axis=0)

# 3. Normalization
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

# 4. SVD on normalized data
U, S, V = np.linalg.svd(norm_data.T, full_matrices=False)
print("✅ SVD complete!")

# 5. Project to Latent Space
Dreduced = np.matmul(U.T, norm_data.T)
Dreduced -= np.mean(Dreduced, axis=1, keepdims=True)
print("✅ Dreduced shape:", Dreduced.shape)

# 6. Save Outputs
np.save(save_dir + "colon_svd_latent.npy", Dreduced)
np.save(save_dir + "colon_svd_latent_labels_raw.npy", labels)

# Encode labels (if needed)
le = LabelEncoder()
encoded_labels = le.fit_transform(labels.flatten())  # Flatten if labels are 2D
np.save(save_dir + "colon_svd_latent_labels.npy", encoded_labels)

print("✅ All steps complete and data saved.")
