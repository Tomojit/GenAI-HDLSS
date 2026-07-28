# Generative Model Implementations

This folder contains the implementations of the generative models used in the comparative experiments presented in the accompanying study on generative AI for high-dimensional, low-sample-size (HDLSS) biological data.

The current repository includes implementations for the Variational Autoencoder (VAE) and LSH-GAN models.

## Models

### Variational Autoencoder (VAE)

The VAE implementation provides the representative autoencoder-based generative model used in the empirical comparison. The model is trained directly on the high-dimensional biological data and generates synthetic samples from the learned latent representation.

The corresponding Python script contains the model architecture, training procedure, synthetic data generation, and parameters used in the experiments.

### LSH-GAN

The LSH-GAN implementation is based on the LSH-GAN framework used as the representative GAN-based model in the comparative study. The model combines generative adversarial learning with locality-sensitive hashing (LSH) to construct local subsets of the training data during model training.

The corresponding Python script contains the implementation used to train LSH-GAN and generate synthetic biological samples for the experiments reported in the paper.

## Example Dataset

The `Dataset` directory contains `GDS1615Dataset.csv`, which is provided as a representative HDLSS gene expression dataset for running the model implementations.

Additional datasets analyzed in the paper are publicly available from their respective repositories and are described in the manuscript.

## Computational Environment

The models were executed in Python-based environments using TensorFlow and related scientific computing libraries. Because the implementations may use different software environments, model-specific package and computational requirements are documented with the corresponding scripts or repository documentation.

## Reproducibility

The scripts provided here correspond to the implementations used in the comparative analysis. Model architectures, training parameters, preprocessing procedures, and evaluation settings are described in the accompanying manuscript and within the corresponding source files where applicable.

Users may modify the dataset paths and model parameters in the scripts to apply the implementations to other high-dimensional biological datasets.
