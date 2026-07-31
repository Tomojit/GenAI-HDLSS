import pandas as pd
import numpy as np
import torch
import anndata
from VAE_model import VAE

# ── Load original data to get mu, std, probe_ids, sample_ids ──────
def loadBreastCancerData(dataFile):
    df = pd.read_csv(dataFile, delimiter='\t', header=0, low_memory=False)
    probe_ids = df['ID'].values
    sample_ids = df.columns[1:]
    trData = df.values[:, 1:].astype('float64')
    trData = trData.T
    L = np.hstack((np.zeros(18), np.ones(18)))
    return trData, L, probe_ids, sample_ids

def standardizeData(data):
    std = np.std(data, axis=0)
    mu = np.mean(data, axis=0)
    std[np.where(std == 0)[0]] = 1.0
    return mu, std, (data - mu) / std

def unStandardizeData(data, mu, std):
    return std * data + mu

dataFile = 'data/BreastCancer/SampleSignals.csv'
X_, L, probe_ids, sample_ids = loadBreastCancerData(dataFile)
mu, std, X = standardizeData(X_)

# ── Decode latent samples ──────────────────────────────────────────
def decode_seed(seed_num):
    npz_path = f"output/scDiffusion/breastCancer_bio/breastCancer_bio_seed{seed_num}/samples.npz"
    vae_path = "output/VAE/breastCancer_bio/model_seed=42_step=99999.pt"

    data = np.load(npz_path)
    latent = data[data.files[0]]
    print(f"Seed {seed_num} latent shape: {latent.shape}")

    autoencoder = VAE(num_genes=22215, device='cuda', seed=0, loss_ae='mse', hidden_dim=128, decoder_activation='ReLU')
    autoencoder.load_state_dict(torch.load(vae_path, map_location='cuda'))
    autoencoder = autoencoder.cuda()
    autoencoder.eval()

    with torch.no_grad():
        latent_tensor = torch.tensor(latent, dtype=torch.float32).cuda()
        decoded = autoencoder(latent_tensor, return_decoded=True)
        decoded = decoded.cpu().numpy()

    # Unstandardize back to original gene space
    decoded = unStandardizeData(decoded, mu, std)

    # Transpose so genes are rows, samples are columns
    decoded = decoded.T

    # Save with probe_ids as rows and sample_ids as columns
    df = pd.DataFrame(decoded, index=probe_ids, columns=sample_ids)
    df.index.name = 'ID'

    out_path = f"output/scDiffusion/breastCancer_bio/breastCancer_bio_synthetic_seed{seed_num}.csv"
    df.to_csv(out_path)
    print(f"Saved: {out_path}, shape: {df.shape}")

for s in range(1, 6):
    decode_seed(s)