# scDiffusion_breastcancer_bio.py
import pandas as pd
import numpy as np
import torch
import anndata
import os
from VAE_model import VAE as scDiffVAE

# ── 1. Load data exactly as prof does ──────────────────────────────
def loadBreastCancerData(dataFile):
    df = pd.read_csv(dataFile, delimiter='\t', header=0, low_memory=False)
    probe_ids = df['ID'].values
    sample_ids = df.columns[1:]
    trData = df.values[:, 1:].astype('float64')
    trData = trData.T  # (36, num_genes)
    L = np.hstack((np.zeros(18), np.ones(18)))
    return trData, L, probe_ids, sample_ids

def standardizeData(data):
    std = np.std(data, axis=0)
    mu = np.mean(data, axis=0)
    std[np.where(std == 0)[0]] = 1.0
    return mu, std, (data - mu) / std

def unStandardizeData(data, mu, std):
    return std * data + mu

# ── 2. Load and standardize ────────────────────────────────────────
# dataFile = '/home/ghoshstudents/VijayNewbase/tghosh-vijay-ConvexLDA/scDiffusion/datah5ad/SampleSignals.csv'
dataFile = '/home/ghoshstudents/VijayNewbase/tghosh-vijay-ConvexLDA/scDiffusion/data/BreastCancer/SampleSignals.csv'
X_, L, probe_ids, sample_ids = loadBreastCancerData(dataFile)
mu, std, X = standardizeData(X_)
print(f"Data shape: {X.shape}")  # (36, num_genes)

num_genes = X.shape[1]
num_samples = X.shape[0]

# ── 3. Save as h5ad for scDiffusion ───────────────────────────────
adata = anndata.AnnData(X=X.astype("float32"))
adata.obs_names = [str(s) for s in sample_ids]
adata.var_names = [str(p) for p in probe_ids]
adata.obs["celltype"] = L.astype(int)
adata.write("datah5ad/breastCancer_bio.h5ad")
print("Saved h5ad")