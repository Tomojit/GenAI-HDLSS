import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pdb

# === CONFIG ===
normalization_method = "zscore"  # options: "log1p", "zscore", "minmax"
loc = "data/RNASeq/RNA.csv"   # Uploaded file path
save_prefix = "data/RNA_svd"            # Where to save outputs
# ==============

# 1. Load dataset
df = pd.read_csv(loc, low_memory=False)

# 2. Extract labels
labels = df["Phenotype/Label"].to_numpy()

# 3. Drop unwanted columns
df = df.drop(["Sample_ID", "Phenotype/Label", "Gender"], axis=1)

# Now df only has gene expressions (ID_REF_1 to ID_REF_6658)
data_only = df.to_numpy().astype(float)

# 4. Mean subtraction (column-wise centering)
data_only -= np.mean(data_only, axis=0)

# 5. Normalization
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
latent = norm_data.T  # shape: (6658, 15)
num_samples = latent.shape[0]
num_classes = 3  # change as needed

# Randomly assign one of the classes to each sample
fake_labels = np.random.randint(0, num_classes, size=num_samples)



# 6. Perform SVD
# U, S, V = np.linalg.svd(norm_data.T, full_matrices=False)
# print("✅ SVD complete!")

# # 7. Project data into latent space
# Dreduced = np.matmul(U.T, norm_data.T)
# Dreduced -= np.mean(Dreduced, axis=1, keepdims=True)

# print("✅ Dreduced shape:", Dreduced.shape)
# pdb.set_trace()

# 8. Save outputs
np.save(f"{save_prefix}_latent.npy", norm_data.T)
np.save(f"{save_prefix}_labels_raw.npy", fake_labels)

# Encode labels and save
le = LabelEncoder()
encoded_labels = le.fit_transform(fake_labels)
# pdb.set_trace()
np.save(f"{save_prefix}_latent_labels.npy", encoded_labels)

print("✅ All steps complete and data saved successfully.")
