import pandas as pd
import matplotlib.pyplot as plt

path = "/home/ghoshstudents/VijayNewbase/tghosh-vijay-ConvexLDA/scDiffusion/output/checkpoint/backbone/suzuki_SVD_diffusion/losses_per_step.csv"

# Skip the first broken row
df = pd.read_csv(path, skiprows=1)

# Keep only required columns
df = df[["step", "mse"]]

# Convert to numeric (safety)
df["step"] = pd.to_numeric(df["step"])
df["mse"] = pd.to_numeric(df["mse"])
df_sampled = df[df["step"] % 500 == 0]
# Plot
plt.figure()
plt.plot(df_sampled["step"], df_sampled["mse"])
plt.xlabel("Step")
plt.ylabel("MSE")
plt.title("Diffusion Training MSE, T 1000")
plt.savefig("Suzuki_mse20kT1000.png")