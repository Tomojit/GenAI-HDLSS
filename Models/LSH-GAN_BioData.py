#!/usr/bin/env python
# coding: utf-8
import pdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from numpy import genfromtxt
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
import timeit
import pandas as pd
import seaborn as sns
import time

# FIXED: Proper GPU configuration for TensorFlow 2.x
print("TensorFlow version:", tf.__version__)

import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {[gpu.name for gpu in gpus]}")
        use_gpu = True
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
        use_gpu = False
else:
    print("No GPU detected, using CPU")
    use_gpu = False

tf.config.optimizer.set_jit(False)  # Disable XLA if it causes issues
tf.config.run_functions_eagerly(False)  # For GPU speed, use False

# FIXED: Disable XLA JIT compilation to avoid libdevice issues
tf.config.optimizer.set_jit(False)

# Use eager execution for better stability
tf.config.run_functions_eagerly(True)
print("Running in eager mode for maximum compatibility")


# LSH Implementation to replace LSHForest
class LSHIndex:
    def __init__(self, n_estimators=10, n_components=64, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.projector = GaussianRandomProjection(
            n_components=n_components,
            random_state=random_state
        )
        self.index = None
        self.fitted_data = None

    def fit(self, X):
        # Convert sparse matrix to dense if needed
        if sparse.issparse(X):
            X = X.toarray()

        # Store the original data
        self.fitted_data = X

        # Project to lower dimension for LSH
        X_projected = self.projector.fit_transform(X)

        # Use NearestNeighbors for approximate search
        self.index = NearestNeighbors(
            # I have changed the kNN value from 5 to 3
            n_neighbors=5,
            algorithm='auto',
            metric='euclidean'
        )
        self.index.fit(X_projected)
        return self

    def kneighbors(self, X, n_neighbors=5):
        if sparse.issparse(X):
            X = X.toarray()

        # Project query points
        X_projected = self.projector.transform(X)

        # Find neighbors in projected space
        distances, indices = self.index.kneighbors(X_projected, n_neighbors=n_neighbors)
        return distances, indices


def chunk(a, n):
    k, m = divmod(len(a), n)
    return [
        tuple(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)])
        for i in range(n)
    ]


# FIXED: New LSH Processor class to avoid multiprocessing issues
class LSHProcessor:
    """Encapsulate LSH processing to avoid global variable issues"""
    def __init__(self, data, obs, n_processes=4):
        print("LSH: Initializing LSH index...")
        self.lshf = LSHIndex(n_estimators=10, random_state=42)
        self.lshf.fit(sparse.coo_matrix(data))
        self.data = data
        self.obs = obs
        self.n_processes = min(n_processes, 8)  # Limit max processes
        print("LSH: Index fitted successfully")
        
    def knn_batch(self, q_indices):
        """Process a batch of queries"""
        try:
            distances, indices = self.lshf.kneighbors(
                # I have changed the kNN value from 5 to 3
                self.data[q_indices, :], 
                n_neighbors=5
            )
            return indices
        except Exception as e:
            print(f"Error in knn_batch: {e}")
            return np.zeros((len(q_indices), 5), dtype=int)
    
    def process(self):
        """Main processing function using ThreadPoolExecutor instead of Pool"""
        print("LSH: Starting LSH process...")
        
        # Split work into chunks
        query_sets = chunk(range(self.obs), self.n_processes)
        print(f"LSH: Split into {len(query_sets)} chunks for parallel processing")
        
        # Use ThreadPoolExecutor instead of multiprocessing.Pool
        # This avoids serialization issues with complex objects
        NN_set = []
        with ThreadPoolExecutor(max_workers=self.n_processes) as executor:
            futures = [executor.submit(self.knn_batch, q_set) for q_set in query_sets]
            
            # Process with timeout
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per chunk
                    NN_set.append(result)
                    print(f"LSH: Completed chunk {i+1}/{len(query_sets)}")
                except TimeoutError:
                    print(f"LSH: Chunk {i+1} timed out, using fallback")
                    NN_set.append(np.zeros((len(query_sets[i]), 5), dtype=int))
                except Exception as e:
                    print(f"LSH: Error in chunk {i+1}: {e}")
                    NN_set.append(np.zeros((len(query_sets[i]), 5), dtype=int))
        
        print("LSH: Parallel processing completed")
        indices = np.vstack(NN_set)
        arr1 = np.ones(self.obs)
        Nb = np.zeros(4)
        m = np.zeros(4)

        for i in range(0, self.obs):
            if arr1[i] != 0:
                Nb = indices[i][1:5].astype(int)
                arr1[Nb] = m
        
        print("LSH: Sampling completed")
        return arr1

def standardizeData(data,mu=None,std=None):
    #data: a m x n matrix where m is the no of observations and n is no of features
    #if any(mu) == None and any(std) == None:
    if mu is None or std is None:
        #pdb.set_trace()
        std = np.std(data,axis=0)
        mu = np.mean(data,axis=0)
        std[np.where(std==0)[0]] = 1.0 #This is for the constant features.
        standardizeData = (data - mu)/std
        return mu,std,standardizeData
    else:
        standardizeData = (data - mu)/std
        return standardizeData

def unStandardizeData(data,mu,std):
    return std * data + mu
    
def load_IBD_Data():
    dataFile = '../Dataset/GDS1615Dataset.csv'
    df = pd.read_csv(dataFile,delimiter=',',header=0,low_memory=False)
    data = df.values[:-1,2:].astype('float64')
    
    return data.T

# Function to create PCA plots
def plot_pca_comparison(original_data, synthetic_data, title_suffix="", save_path=None):
    """
    Create clean PCA visualization comparing original and synthetic data
    """
    # Fit PCA on original data first
    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(original_data)
    synthetic_pca = pca.transform(synthetic_data)

    # Save PCA coordinates to CSV files
    if title_suffix:
        # Clean the title suffix for filename (remove special characters)
        clean_suffix = title_suffix.replace("(", "").replace(")", "").replace(" ", "_")
        original_filename = f"original_pca_coordinates{clean_suffix}.csv"
        synthetic_filename = f"synthetic_pca_coordinates{clean_suffix}.csv"
    else:
        original_filename = "original_pca_coordinates.csv"
        synthetic_filename = "synthetic_pca_coordinates.csv"

    # Save original PCA coordinates
    original_pca_df = pd.DataFrame(original_pca, columns=['PC1', 'PC2'])
    original_pca_df.to_csv(original_filename, index=False)

    # Save synthetic PCA coordinates
    synthetic_pca_df = pd.DataFrame(synthetic_pca, columns=['PC1', 'PC2'])
    synthetic_pca_df.to_csv(synthetic_filename, index=False)

    print(f"Saved original PCA coordinates to: {original_filename}")
    print(f"Saved synthetic PCA coordinates to: {synthetic_filename}")

    # Create the enhanced PCA plot
    plt.figure(figsize=(12, 8))
    plt.scatter(
        original_pca[:, 0], original_pca[:, 1],
        label="Real Samples", c='blue', alpha=0.6, s=30
    )
    plt.scatter(
        synthetic_pca[:, 0], synthetic_pca[:, 1],
        label="Synthetic Samples", c='red', alpha=0.6, s=30
    )
    plt.title(f"GAN - PCA Visualization of Real vs Synthetic Data {title_suffix}")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    # Print explained variance
    print(f"PCA Explained Variance {title_suffix}:")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"Total: {pca.explained_variance_ratio_[:2].sum():.1%}")
    print(f"Real data shape: {original_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print("-" * 50)
    return pca, original_pca, synthetic_pca


# Load data with proper type handling
try:   
    # Load the IBD data set
    X = load_IBD_Data()
    sampleCnt = len(X)

    print(f"Transposed features shape (samples x genes): {X.shape}")

    # Data validation and preparation
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        print(f"Found {nan_count} NaN values in features. Filling with zeros.")
        X = np.nan_to_num(X, nan=0.0)

    # If your data is sparse matrix, convert to dense
    if hasattr(X, 'toarray'):
        print("Converting sparse matrix to dense array")
        X = X.toarray()

    # Normalize the data (standardization: mean=0, std=1)
    print("Normalizing data...")

    mu,std,X = standardizeData(X)
    data = X
    
    print("Data normalization completed successfully")

except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# CHANGED: Update variable assignments to match new data orientation
x_plot = data
Xnew = X

p = sampleCnt
# Note: row now represents number of genes, col represents number of samples
row = x_plot.shape[0]  # Number of genes
col = x_plot.shape[1]  # Number of samples

# Store original data for PCA comparison
original_data = X

# Number of iteration itr=1 for all except klein. itr=2 for klein
itr = 1

# FIXED: Use the new LSHProcessor class with threading
print(f"Starting LSH sampling iterations: {itr}")
for i in range(0, itr):
    print(f"\n{'='*60}")
    print(f"LSH iteration {i+1}/{itr}")
    print(f"{'='*60}")
    rowlsh = Xnew.shape[0]  # Number of samples
    print(f"Processing {rowlsh} samples with LSH...")
    
    try:
        # Use the new LSHProcessor class with 4 threads
        processor = LSHProcessor(Xnew, rowlsh, n_processes=4)
        result = processor.process()
        
        c = np.nonzero(result)
        c1 = c[0]
        Xnew = Xnew[c1, :]  # Select subset of genes
        print(f"After LSH iteration {i+1}: {Xnew.shape[0]} samples remaining")
    except Exception as e:
        print(f"Error in LSH iteration {i+1}: {e}")
        print("Continuing with current Xnew...")
        import traceback
        traceback.print_exc()
        break

print(f"\nFinal Xnew shape after LSH sampling: {Xnew.shape}")
print(f"Final data: {Xnew.shape[0]} samples x {Xnew.shape[1]} genes")


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# FIXED: TensorFlow 2.x compatible model definitions
# Note: Generator output_dim should match number of samples (columns)
class Generator(tf.keras.Model):
    def __init__(self, output_dim, hidden_sizes=[16, 16]):
        super(Generator, self).__init__()
        self.hidden_layers = []
        for size in hidden_sizes:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, z):
        x = z
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class Discriminator(tf.keras.Model):
    def __init__(self, input_dim, hidden_sizes=[16, 16]):
        super(Discriminator, self).__init__()
        self.hidden_layers = []
        for size in hidden_sizes:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        self.feature_layer = tf.keras.layers.Dense(input_dim)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        features = self.feature_layer(x)
        output = self.output_layer(features)
        return output, features


# Initialize models
# CHANGED: Generator output_dim should be number of samples (columns)
generator = Generator(col)  # col = number of samples
discriminator = Discriminator(col)  # input_dim = number of samples

# FIXED: TensorFlow 2.x optimizers
gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# FIXED: TensorFlow 2.x loss function
binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Simpler train_step without @tf.function decorator for better compatibility
def train_step(real_data, noise):
    """Training step function (eager mode for compatibility)"""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake data
        fake_data = generator(noise, training=True)

        # Get discriminator outputs
        real_output, real_features = discriminator(real_data, training=True)
        fake_output, fake_features = discriminator(fake_data, training=True)

        # Calculate losses
        real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
        fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = real_loss + fake_loss
        gen_loss = binary_crossentropy(tf.ones_like(fake_output), fake_output)

        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Apply gradients
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return gen_loss, disc_loss


# Training parameters
nd_steps = 10
ng_steps = 10
epochs = 100

# Loss tracking
loss_records = []

start = timeit.default_timer()

print("\n" + "="*60)
print("Starting GAN training...")
print(f"Using device: {'GPU' if use_gpu else 'CPU'}")
print(f"Data shape: {x_plot.shape[0]} samples x {x_plot.shape[1]} genes")
print("="*60 + "\n")

# Training loop with periodic saving
for epoch in range(1, epochs + 1):
    X_batch = tf.constant(x_plot, dtype=tf.float32)
    row1 = row - Xnew.shape[0]  # Number of additional genes needed
    da1 = sample_Z(row1, col)   # Generate random genes
    Z_batch = tf.constant(np.vstack((da1, Xnew)), dtype=tf.float32)

    # Training steps
    for _ in range(nd_steps):
        gen_loss, disc_loss = train_step(X_batch, Z_batch)

    print(f"Epoch {epoch}/{epochs} \t Discriminator loss: {disc_loss.numpy():.4f} \t Generator loss: {gen_loss.numpy():.4f}")

    # Store loss values
    loss_records.append([epoch, disc_loss.numpy(), gen_loss.numpy()])

    # Save outputs every 10 epochs
    if epoch % 100 == 0:
        print(f"\n=== Saving outputs at Epoch {epoch} ===")

        # Generate synthetic data
        synthetic_sample = generator(Z_batch, training=False).numpy()

        # Save synthetic data (CSV)
        pd.DataFrame(synthetic_sample).to_csv(f"LSH-GAN_synthetic_data_epoch_{epoch}.csv", index=False)

        # Save generator weights (CSV)
        gen_weights = []
        for var in generator.trainable_variables:
            gen_weights.extend(var.numpy().flatten())
        pd.DataFrame(gen_weights).to_csv(f"generator_epoch_{epoch}.csv", index=False, header=["weights"])

        # Save discriminator weights (CSV)
        disc_weights = []
        for var in discriminator.trainable_variables:
            disc_weights.extend(var.numpy().flatten())
        pd.DataFrame(disc_weights).to_csv(f"discriminator_epoch_{epoch}.csv", index=False, header=["weights"])

        # Combine both datasets
        combined_data = np.vstack([original_data, synthetic_sample])

        # Perform PCA on the combined dataset
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_data)

        # Split the PCA results back to original and synthetic
        n_original = len(original_data)
        original_pca = combined_pca[:n_original]
        synthetic_pca = combined_pca[n_original:]

        # Save PCA coordinates
        pd.DataFrame(original_pca, columns=['PC1', 'PC2']).to_csv(f"original_pca_coordinates_epoch_{epoch}.csv", index=False)
        pd.DataFrame(synthetic_pca, columns=['PC1', 'PC2']).to_csv(f"synthetic_pca_coordinates_epoch_{epoch}.csv", index=False)

        # PCA plot
        plt.figure(figsize=(10, 7))
        plt.scatter(original_pca[:, 0], original_pca[:, 1], label="Real", alpha=0.6, c="blue")
        plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], label="Synthetic", alpha=0.6, c="red")
        plt.title(f"PCA Comparison (Epoch {epoch})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"GAN_PCA_epoch_{epoch}.png", dpi=300)
        plt.close()

        # Save loss plot every 10 epochs
        loss_df = pd.DataFrame(loss_records, columns=["Epoch", "Discriminator_Loss", "Generator_Loss"])
        plt.figure(figsize=(10, 6))
        plt.plot(loss_df["Epoch"], loss_df["Discriminator_Loss"], label="Discriminator Loss", color="blue")
        plt.plot(loss_df["Epoch"], loss_df["Generator_Loss"], label="Generator Loss", color="red")
        plt.title(f"GAN Training Losses (Up to Epoch {epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"training_losses_epoch_{epoch}.png", dpi=300)
        plt.close()
        
        print(f"Saved outputs for epoch {epoch}\n")

# Save all loss records to CSV once at the end
loss_df = pd.DataFrame(loss_records, columns=["Epoch", "Discriminator_Loss", "Generator_Loss"])
loss_df.to_csv("training_losses.csv", index=False)

stop = timeit.default_timer()
print('\n' + "="*60)
print(f'Training Time: {stop - start:.2f} seconds')
print("Training completed successfully! CSV + PNG files saved.")
print("="*60)
