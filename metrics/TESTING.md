# Testing and Metric Collection

This directory contains the tools used to evaluate the performance and quality of different image sorting methods. The testing framework allows for automated parameter sweeps across multiple sorting algorithms to determine the optimal configuration for a given dataset.

## Overview

The testing system consists of two primary components:
1. **Test Rig (`test_rig.py`)**: A CLI tool that manages experiment execution, parameter sweeps, and result logging.
2. **Evaluator (`evaluator.py`)**: The core logic that runs a specific sorter configuration in "headless" mode (without moving files) and calculates quantitative metrics.

## How to Run Tests

The `test_rig.py` script uses the `click` library for its command-line interface.

### Example Commands

**Run all configured experiments:**
```bash
python metrics/test_rig.py
```

**Run specific sorters (comma-separated):**
```bash
python metrics/test_rig.py --sorters k-means,exposure,histogram_3d
```

**Specify a custom input directory:**
```bash
python metrics/test_rig.py --input /path/to/your/images
```

**Process RAW files instead of JPGs:**
```bash
python metrics/test_rig.py --raw
```

**Save results to a specific CSV file:**
```bash
python metrics/test_rig.py --output-csv my_experiment_results.csv
```

## Metric Definitions

The following metrics are collected for every parameter combination:

| Metric | Description | Ideal Value |
| :--- | :--- | :--- |
| **Silhouette Score** | Measures how similar an image is to its own cluster compared to other clusters. | Higher (closer to 1.0) |
| **Davies-Bouldin Index** | The average similarity measure of each cluster with its most similar cluster. | Lower (closer to 0.0) |
| **Balance Std Dev** | The standard deviation of the number of images per cluster. | Lower (indicates more uniform distribution) |
| **Duration** | Total time taken for feature extraction and clustering. | Lower |
| **Num Clusters** | The actual number of clusters formed by the algorithm. | Varies by goal |
| **Images Processed** | Number of images that were successfully analyzed. | Equal to total input images |

## Extending the Framework

### Adding a New Sorter
To add a new sorting method to the evaluation suite:
1. Open `metrics/evaluator.py`.
2. Add the sorter's `Extractor` and `Clusterer` classes to the `SorterRegistry.REGISTRY` dictionary.
3. Open `metrics/test_rig.py`.
4. Add a new entry to the `EXPERIMENTS` dictionary defining the parameter sweeps you wish to perform (e.g., different values for `clusters`, `bins`, or `threshold`).

### Modifying Parameters
Parameters defined in `EXPERIMENTS` are passed to the evaluator, which uses heuristics to distribute them between the `Extractor` and the `Clusterer`. Common parameter names include:
- `k_colors`: Used by color extractors.
- `bins`: Used by histogram extractors.
- `clusters`: Maps to `n_clusters` for the clusterer.
- `threshold`: Used by distance-based or hash-based clusterers.