# Color-Sorter CLI

A modular Python tool that automatically organizes images (JPG or RAW) into "buckets" based on their visual characteristics. The tool is designed to be format-agnostic, allowing for efficient sorting of pre-rendered JPGs or high-fidelity RAW files with sidecar edits.

## Features

- **Dual-Format Support**: Seamlessly switch between processing standard JPG/JPEG images and professional RAW formats (ARW, CR2, NEF, DNG).
- **XMP-Aware RAW Rendering**: For RAW files, the tool respects `.xmp` sidecar edits by attempting to render them via a tiered pipeline before analysis.
- **Perceptual Color Analysis**: Converts images to CIELAB color space for accurate visual distance calculations.
- **Automated Clustering**: Uses K-Means and Agglomerative Clustering to group images with similar palettes or structures.
- **Modular Architecture**: A decoupled framework separates the sorting logic (feature extraction and clustering) from the image ingestion layer.
- **High Performance**: Utilizes multiprocessing for parallel image loading and analysis.
- **Rigorous Evaluation**: Includes a built-in testing suite that uses a rank-based composite scoring system (Borda Count) to objectively determine the best sorting strategy for a given dataset.

## Prerequisites

### System Requirements
- **Python 3.10+**
- **Operating System**: Linux, macOS, or Windows.

### Optional Dependencies (for RAW High Fidelity)
To ensure that `.xmp` sidecar edits are fully applied during RAW analysis, it is highly recommended to install **darktable**. The script will automatically detect `darktable-cli` and use it for rendering.

- **Linux**: `sudo apt install darktable`
- **macOS**: `brew install darktable`
- **Windows**: Install via the official darktable website and ensure `darktable-cli` is in your system PATH.

*If `darktable-cli` is not found, the tool falls back to extracting the embedded JPEG preview or performing basic demosaicing via `rawpy`.*

## Installation

1. Clone this repository or download the script.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   .\venv\Scripts\activate   # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Sorting Methods

The tool provides several specialized sorters, each using a different feature extraction and clustering strategy:

### 1. Exposure Sorter (`exposure_sorter.py`)
- **Method**: Analyzes the Luminance (L) channel of the LAB color space to calculate mean brightness and standard deviation.
- **Best For**: Sorting by lighting conditions (e.g., Night, Golden Hour, Daytime, Overexposed).
- **Note**: Currently the top-performing method for balanced, perceptually useful sorting.

### 2. K-Means Palette Sorter (`k-means_sorter.py`)
- **Method**: Extracts the top $K$ dominant colors from each image in CIELAB space using MiniBatchKMeans.
- **Best For**: Grouping images by their primary color schemes (e.g., "all the blue-toned photos").
- **Note**: Provides extreme internal cohesion but can produce imbalanced cluster sizes.

### 3. Histogram Sorter (`histogram_sorter.py`)
- **Method**: Computes normalized color histograms for Hue, Saturation, and Value (HSV) channels.
- **Best For**: Grouping images with similar overall color distributions, regardless of where the colors are located.
- **Feature**: Uses Cosine similarity via Agglomerative Clustering for high-accuracy distribution matching.


### 4. Perceptual Hash Sorter (`hash_sorter.py`)
- **Method**: Generates a visual "fingerprint" of the image using one of three hashing algorithms:
  - **aHash (Average Hash)**: Fast, based on average pixel intensity.
  - **dHash (Difference Hash)**: Tracks gradients between adjacent pixels.
  - **pHash (Perceptual Hash)**: Uses Discrete Cosine Transform (DCT) to focus on low-frequency structures.
- **Best For**: Finding near-duplicate images or images with very similar compositions.
- **Feature**: Uses Hamming Distance to cluster binary hashes.

### 5. CV Feature Sorter (`cv_feature_sorter.py`)
- **Method**: A composite approach combining multiple computer vision metrics:
  - **ORB Spatial Grid**: Distribution of keypoints across the image.
  - **HSV Histogram**: General color distribution.
  - **Laplacian Variance**: Measure of image sharpness/blur.
  - **Canny Edge Density**: Measure of geometric complexity.
- **Best For**: Complex sorting based on a mix of texture, sharpness, and color.
- **Feature**: Uses a `ScalingKMeansClusterer` to standardize diverse feature scales.

## Modularity & Expansion

The tool is built on a modular framework (`framework.py`) that allows developers to easily add new sorting strategies without modifying the core orchestration logic.

### The Framework Architecture
The `SortingFramework` class requires three components:
1. **`BaseImageLoader`**: Handles how images are read (e.g., `JPGLoader`, `RAWLoader`).
2. **`BaseFeatureExtractor`**: Defines how to turn an image into a numerical vector (the "what" of the sort).
3. **`BaseClusterer`**: Defines how to group those vectors into buckets (the "how" of the sort).

### How to Expand
To create a new sorter:
1. **Implement a new `BaseFeatureExtractor`**: Define the `extract()` method to return a numpy array.
2. **Implement a new `BaseClusterer`** (Optional): If the existing `KMeansClusterer` or `AgglomerativeClusterer` doesn't fit your needs.
3. **Create a CLI script**: Use the `run_sorting_pipeline` helper from `framework.py` to connect your extractor and clusterer to the CLI.

## Usage

The tool is executed via the command line. It supports a flexible ingestion system with sensible defaults.

### Common Options

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `./jpg_storage` or `./raw_storage` | Path to directory containing images. |
| `--output-dir` | | `./output_buckets` | Directory where sorted buckets will be created. |
| `--move` | | `False` | Delete source files after they have been copied to buckets. |
| `--raw` | | `False` | Process RAW files instead of JPGs. |
| `--verbose` | `-v` | `False` | Enable debug logging. |

### Examples & Testing

You can test the different sorting methods using the following commands:

**Sort by lighting/exposure (Recommended):**
```bash
python3 exposure_sorter.py --input ./jpg_storage --output-dir ./output_exposure_test -c 20
```

**Sort by dominant color palettes (K-Means):**
```bash
python3 k-means_sorter.py --input ./jpg_storage --output-dir ./output_k-means_test -k 2
```

**Sort by color distribution (Histogram):**
```bash
python3 histogram_sorter.py --input ./jpg_storage --output-dir ./output_histogram_test --threshold 0.45
```

**Sort by visual structure (Perceptual Hashing - pHash):**
```bash
python3 hash_sorter.py --input ./jpg_storage --method phash --output-dir ./output_phash_test --threshold 0.45
```

**Sort by composite CV features:**
```bash
python3 cv_feature_sorter.py --input ./jpg_storage --output-dir ./output_cv_test -c 20
```
