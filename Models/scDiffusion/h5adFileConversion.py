import scanpy as sc
import numpy as np

# # === Paths ===
# input_path = "/home/prathyusha/SaiVijay/Vijay_ConvexLDA/scDiffusionCode/data/tabula_muris/all.h5ad"
# output_dir = "/home/prathyusha/SaiVijay/Vijay_ConvexLDA/scDiffusionCode/data/"
# samples_path = f"{output_dir}muris_50_samples.npy"
# labels_path = f"{output_dir}muris_50_labels.npy"

# # === Load dataset ===
# adata = sc.read_h5ad(input_path)
# print("✅ Loaded .h5ad file with shape:", adata.shape)

# # === Auto-detect label column ===
# possible_labels = ['cell_type', 'cell_ontology_class', 'celltype', 'label', 'cell_label']
# label_column = None
# for col in possible_labels:
#     if col in adata.obs.columns:
#         label_column = col
#         print(f"✅ Found label column: '{label_column}'")
#         break

# if label_column is None:
#     raise ValueError("❌ No known label column found in adata.obs.")

# # === Randomly sample 50 cells ===
# n_samples = 50
# random_indices = np.random.choice(adata.shape[0], n_samples, replace=False)
# adata_subset = adata[random_indices]

# # === Extract expression data and labels ===
# expression_data = adata_subset.X.toarray() if hasattr(adata_subset.X, 'toarray') else adata_subset.X
# labels = adata_subset.obs[label_column].to_numpy()

# # === Save separately ===
# np.save(samples_path, expression_data)
# np.save(labels_path, labels)

# print(f"\n✅ Saved 50 samples to: {samples_path}")
# print(f"✅ Saved corresponding labels to: {labels_path}")

# from sklearn.preprocessing import LabelEncoder
# labels = np.load("data/muris_50_labels.npy", allow_pickle=True)
# le = LabelEncoder()
# encoded_labels = le.fit_transform(labels)
# np.save("data/muris_50_labels.npy", encoded_labels)
# print("✅ Encoded labels saved.")
# print("📋 Label classes:", le.classes_)

# import scanpy as sc
# import numpy as np

# # Load original h5ad file
# input_path = "data/tabula_muris/all.h5ad"  # change if needed
# adata = sc.read_h5ad(input_path)
# print("Original shape:", adata.shape)

# # Randomly select 50 samples
# np.random.seed(42)  # for reproducibility
# selected_indices = np.random.choice(adata.shape[0], 50, replace=False)
# adata_subset = adata[selected_indices]

# # Save new h5ad file with only 50 samples
# output_path = "data/tabula_muris/muris_50_subset.h5ad"
# adata_subset.write(output_path)

# print(f"✅ Saved 50-sample subset to: {output_path}")

# import scanpy as sc
# import pandas as pd

# # Load your full 50-cell data
# adata = sc.read_h5ad("data/tabula_muris/muris_50_subset.h5ad")

# # Load pretrained gene order
# gene_order = pd.read_csv("models/annotation_model_v1/gene_order.tsv", header=None)[0].tolist()

# # Filter genes
# filtered_genes = [g for g in gene_order if g in adata.var_names]
# adata_filtered = adata[:, filtered_genes]

# # Save it
# adata_filtered.write("data/tabula_muris/muris_50_filtered.h5ad")
# print(f"✅ Saved filtered file with shape: {adata_filtered.shape}")

import scanpy as sc
import pandas as pd

# Load your full 50-cell data
adata = sc.read_h5ad("data/tabula_muris/muris_50_subset.h5ad")

# Load pretrained gene order
gene_order = pd.read_csv("models/annotation_model_v1/gene_order.tsv", header=None)[0].tolist()

# Filter genes
filtered_genes = [g for g in gene_order if g in adata.var_names]
adata_filtered = adata[:, filtered_genes]

# Save it
adata_filtered.write("data/tabula_muris/muris_50_filtered.h5ad")
print(f"✅ Saved filtered file with shape: {adata_filtered.shape}")

