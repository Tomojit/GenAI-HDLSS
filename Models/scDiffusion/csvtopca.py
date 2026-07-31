import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# load
real = np.loadtxt("real.csv", delimiter=",")
syn  = np.loadtxt("syn.csv", delimiter=",")

print("real shape:", real.shape)
print("syn shape :", syn.shape)

# combine
X = np.vstack([real, syn])

# PCA to 2D
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

# split back
# real_2d = X2[:real.shape[0]]
# syn_2d  = X2[real.shape[0]:]


real_2d = real
syn_2d  = syn

# plot
plt.figure(figsize=(7,6))
plt.scatter(real_2d[:,0], real_2d[:,1], label="Real", alpha=0.7)
plt.scatter(syn_2d[:,0], syn_2d[:,1], label="Synthetic", alpha=0.7)

plt.legend()
plt.title("Real vs Synthetic (PCA) 10K(T) 2Million(epoches)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
plt.savefig("mossdFullcorruptsampling5.png")