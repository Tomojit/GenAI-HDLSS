import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load original data
def loadBreastCancerData(dataFile):
    df = pd.read_csv(dataFile, delimiter='\t', header=0, low_memory=False)
    trData = df.values[:, 1:].astype('float64')
    trData = trData.T
    return trData

real = loadBreastCancerData('data/BreastCancer/SampleSignals.csv')

fig, axes = plt.subplots(1, 5, figsize=(30, 6))

for seed in range(1, 6):
    syn_df = pd.read_csv(f"output/scDiffusion/breastCancer_bio/breastCancer_bio_synthetic_seed{seed}.csv", index_col=0)
    synthetic = syn_df.values.T  # transpose back to samples x genes

    combined = np.vstack([real, synthetic])
    pca = PCA(n_components=2)
    proj = pca.fit_transform(combined)

    real_proj = proj[:len(real)]
    syn_proj = proj[len(real):]

    ax = axes[seed-1]
    ax.scatter(real_proj[:,0], real_proj[:,1], label="Original Data", alpha=0.7, s=40, c="red")
    ax.scatter(syn_proj[:,0], syn_proj[:,1], label="Synthetic Data", alpha=0.7, s=40, c="blue")
    ax.set_title(f"Seed {seed}", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)

plt.suptitle("Breast Cancer (Bio): Real vs Synthetic", fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig("output/scDiffusion/breastCancer_bio/pca_breastCancer_bio_all_seeds.png", dpi=150)
print("Saved.")