# GDS1615 IBD Dataset

This folder contains the processed gene expression dataset `GDS1615Dataset.csv`, used as a representative high-dimensional, low-sample-size (HDLSS) biological dataset in the comparative experiments reported in this study.

## Dataset

- **File:** `GDS1615Dataset.csv`
- **Dataset:** GDS1615
- **Biological application:** Inflammatory Bowel Disease (IBD)
- **Data type:** Gene expression data
- **Source:** NCBI Gene Expression Omnibus (GEO)
- **GEO accession:** GDS1615

The dataset contains gene expression measurements from samples associated with inflammatory bowel disease. It was used to evaluate representative generative models under HDLSS conditions.

## File Format

`GDS1615Dataset.csv` contains the processed expression matrix used as input to the generative modeling experiments. Rows correspond to biological samples and columns correspond to gene expression features, together with the class information required for the experiments.

## Usage

This dataset is provided as a representative example for reproducing the experimental workflow described in the accompanying paper. The corresponding model implementations and evaluation scripts are available in the other directories of this repository.

## Data Source

The original dataset is publicly available through the NCBI Gene Expression Omnibus under accession **GDS1615**. Users interested in the original data and associated biological and experimental metadata should refer to the GEO record.

## Citation

If this dataset is used in subsequent work, please cite the original study associated with GDS1615 and the NCBI Gene Expression Omnibus.
