# Campolina

A deep-learning framework for accurate segmentation of raw Nanopore sequencing signals.

## Overview

Campolina is a convolution-based neural network model trained to identify event borders in raw Nanopore signals. It supports both CNN and TCN (Temporal Convolutional Network) architectures.

## Tech Stack

- **Language**: Python 3.12
- **Deep Learning**: PyTorch (CPU build), pytorch-tcn
- **Bioinformatics**: pod5, pysam, biopython
- **Data Processing**: numpy, pandas, polars, pyarrow, duckdb
- **Visualization**: matplotlib, plotnine
- **Package Manager**: uv (via Replit)

## Project Structure

```
campolina/
  data/          - POD5 and BAM file loading utilities
  evaluation/    - Segmentation quality assessment pipeline
  groundtruth/   - Ground truth extraction tools
  model/         - Neural network architecture definitions
    model.py     - CNN-based EventDetector
    tcn_model.py - TCN-based TCNEventDetector
    pl_model.py  - PyTorch Lightning wrapper
config_main.py   - Training entry point
inference.py     - Inference script for pre-trained models
train_config.json - Hyperparameter and path configuration
```

## Usage

For inference with a pre-trained model:
```bash
python3 inference.py --help
```

For training:
```bash
python3 config_main.py --help
```

Pre-trained weights can be downloaded from [Zenodo](https://zenodo.org/records/15626806).

## Workflow

The "Start application" workflow verifies the library loads correctly and shows the environment is ready.
