# Color-Sorter CLI

A modular Python tool that automatically organizes images (JPG or RAW) into "buckets" based on their dominant color palettes. The tool is designed to be format-agnostic, allowing for efficient sorting of pre-rendered JPGs or high-fidelity RAW files with sidecar edits.

## Features

- **Dual-Format Support**: Seamlessly switch between processing standard JPG/JPEG images and professional RAW formats (ARW, CR2, NEF, DNG).
- **XMP-Aware RAW Rendering**: For RAW files, the tool respects `.xmp` sidecar edits by attempting to render them via a tiered pipeline before analysis.
- **Perceptual Color Analysis**: Converts images to CIELAB color space for accurate visual distance calculations.
- **Automated Clustering**: Uses K-Means and Agglomerative Clustering to group images with similar palettes.
- **Modular Architecture**: A decoupled framework separates the sorting logic (feature extraction and clustering) from the image ingestion layer.
- **High Performance**: Utilizes multiprocessing for parallel image loading and analysis.

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

## Usage

The tool is executed via the command line. It supports a flexible ingestion system with sensible defaults.

### Basic Command (JPG Mode)
By default, the tool looks for JPG images in `./jpg_storage`.
```bash
python k-means_cluster.py
```

### RAW Mode
To process RAW files (defaulting to `./raw_storage`), use the `--raw` flag.
```bash
python k-means_cluster.py --raw
```

### Common Options

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | `./jpg_storage` or `./raw_storage` | Path to directory containing images. |
| `--output-dir` | | `./output_buckets` | Directory where sorted buckets will be created. |
| `--colors` | `-k` | `3` | Number of dominant colors to extract per image. |
| `--clusters` | | `None` | Force a specific number of target buckets. |
| `--threshold` | | `50.0` | Distance threshold for clustering (used if `--clusters` is not set). |
| `--move` | | `False` | Delete source files after they have been copied to buckets. |
| `--raw` | | `False` | Process RAW files instead of JPGs. |
| `--verbose` | `-v` | `False` | Enable debug logging. |

### Examples

**Sort JPGs from a custom directory into buckets:**
```bash
python k-means_cluster.py -i ./my_vacation_jpgs
```

**Sort RAW files from default storage and move them to buckets:**
```bash
python k-means_cluster.py --raw --move
```

**Force exactly 5 buckets, analyzing 5 dominant colors per image:**
```bash
python k-means_cluster.py --raw -k 5 --clusters 5
```

## How it Works

1. **Modular Ingestion**: 
   - The CLI selects a `Loader` based on the format (JPG or RAW).
   - **JPGLoader**: Directly reads images using OpenCV.
   - **RAWLoader**: Uses a rendering pipeline (`darktable-cli` $\rightarrow$ `rawpy` thumbnail $\rightarrow$ `rawpy` demosaic) to create a temporary RGB representation, respecting XMP sidecars.
2. **Downscaling**: Images are resized to 400px on the long edge to optimize processing speed.
3. **Color Extraction**: 
   - Pixels are converted from sRGB to **CIELAB** space.
   - **K-Means clustering** is performed on each image to find the $K$ most dominant colors.
   - A feature vector is created from these centroids, ordered by pixel frequency.
4. **Global Clustering**: The feature vectors of all images are clustered using **Agglomerative Clustering** (or KMeans), grouping images with similar overall palettes.
5. **Organization**: Files (and their associated sidecars) are copied into folders named by their cluster ID and a human-readable color family (e.g., `bucket_0_warm_gold`).