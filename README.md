# GenAI-HDLSS

This repository contains code, example data, synthetic datasets, evaluation scripts, and literature-search resources associated with the study **"Generative AI for High-Dimensional Biological Data Under HDLSS Constraints: A Critical Review and Comparative Study."**

The repository is intended to support reproducibility of the comparative experiments and quantitative evaluations presented in the study.

## Repository Contents

The repository contains the following main components:

- **Dataset/** - Contains the GDS1615 inflammatory bowel disease (IBD) gene expression dataset used as a representative HDLSS dataset for reproducing the experiments.
- **Models/** - Contains implementations used for the VAE and LSH-GAN experiments.
- **Evaluation/** - Contains the quantitative evaluation script and synthetic datasets generated from five independent runs of each representative model.
- **Literature Search/** - Contains the OpenAlex search script and provenance information used for the literature survey.

Additional README files within individual directories provide information specific to the corresponding data, models, and evaluation resources.

## Running the Models

The scripts use relative paths and should therefore be executed from the directory indicated below.

### VAE

Navigate to the `Models` directory:

```bash
cd GenAI-HDLSS/Models
```

Start IPython:

```bash
ipython3
```

From the IPython prompt, execute the VAE script:

```python
run VAE_BioData.py
```

The provided configuration uses the IBD dataset included in the repository. Dataset paths and experimental parameters can be modified in the script for other datasets or experimental settings.

### LSH-GAN

Navigate to the `Models` directory:

```bash
cd GenAI-HDLSS/Models
```

Start IPython:

```bash
ipython3
```

Then execute:

```python
run LSH-GAN_BioData.py
```

The LSH-GAN implementation used in this study is based on the publicly available implementation associated with the original LSH-GAN work and was adapted for the experiments reported in the accompanying study.

### scDiffusion

The scDiffusion experiments use the implementation provided by the original authors. Instructions and commands required to reproduce the scDiffusion experiments are provided separately in the corresponding repository directory.

## Quantitative Evaluation

Synthetic datasets from five independent training runs are provided for the representative models evaluated on the IBD dataset. These files are stored in compressed format and should be extracted before running the evaluation.

The quantitative evaluation includes:

- Subspace Cosine Similarity (SCS)
- Kullback-Leibler Divergence (KLD)
- Maximum Mean Discrepancy (MMD)
- Wasserstein Distance

To reproduce the evaluation, navigate to the `Evaluation` directory:

```bash
cd GenAI-HDLSS/Evaluation
```

Start IPython:

```bash
ipython3
```

Then run:

```python
run Compute_SCS_KLD_MMD_Wasserstein_Metrics.py
```

The evaluation script compares the empirical IBD data with the synthetic datasets generated across the five independent runs. Model-specific synthetic datasets are provided in the corresponding directories within the repository.

The quantitative results reported in the study are summarized as the mean and standard deviation across the five independent runs.

## Computational Environment

The experiments were conducted using Python-based implementations with TensorFlow and PyTorch, depending on the model.

For LSH-GAN, the computational environment included:

- Python: 3.13.5
- IPython: 8.30.0
- TensorFlow: 2.20.0
- NumPy: 2.1.3
- Pandas: 3.0.3
- Scikit-learn: 1.6.1
- SciPy: 1.15.3
- Matplotlib: 3.10.0
- GPU: NVIDIA Tesla V100-PCIE-16GB
- TensorFlow CUDA: 12.5.1
- cuDNN: 9
- NVIDIA Driver: 580.173.02

Model-specific dependencies and computational requirements are documented with the corresponding implementations where applicable.

## Literature Search Reproducibility

The `Literature Search` directory contains the OpenAlex retrieval script and provenance information used for the literature survey. The search covered journal articles published between 2015 and 2025 related to generative AI methods for high-dimensional biological data.

The complete search methodology and Boolean query are described in the manuscript and Supplementary Material.

## Data

The repository includes `GDS1615Dataset.csv` as a representative HDLSS biological dataset for reproducing the model and evaluation workflow.

Additional datasets analyzed in the study are publicly available from their original repositories and are described in the accompanying manuscript.

## Reproducibility Notes

The scripts were tested after downloading the complete repository. Because the current implementations use relative file paths, model scripts should be executed from the `Models` directory and the quantitative evaluation script should be executed from the `Evaluation` directory as described above.

The synthetic data files are compressed to accommodate repository file-size restrictions. Extract the corresponding archives before running the quantitative evaluation.

## Citation

If you use the code, data-processing workflow, or evaluation resources provided in this repository, please cite the accompanying paper.

Citation information will be updated following publication.
