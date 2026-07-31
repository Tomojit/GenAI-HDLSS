import pandas as pd
import numpy as np
import torch
from VAE_model import VAE

def decode_seed(seed_num, num_genes=6658, hidden_dim=128):
    npz_path = f"output/scDiffusion/RNA/RNA_seed{seed_num}/samples.npz"
    vae_path = "output/VAE/RNA/model_seed=42_step=99999.pt"
    
    data = np.load(npz_path)
    latent = data[data.files[0]]
    print(f"Seed {seed_num} latent shape: {latent.shape}")
    
    autoencoder = VAE(num_genes=num_genes, device='cuda', seed=0, loss_ae='mse', hidden_dim=hidden_dim, decoder_activation='ReLU')
    autoencoder.load_state_dict(torch.load(vae_path, map_location='cuda'))
    autoencoder = autoencoder.cuda()
    autoencoder.eval()
    
    with torch.no_grad():
        latent_tensor = torch.tensor(latent, dtype=torch.float32).cuda()
        decoded = autoencoder(latent_tensor, return_decoded=True)
        decoded = decoded.cpu().numpy()
    
    df = pd.DataFrame(decoded)
    out_path = f"output/scDiffusion/RNA/RNA_synthetic_seed{seed_num}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}, shape: {df.shape}")

for s in range(1, 6):
    decode_seed(s)