# RAW Image Palette Clustering CLI

A standalone Python tool that automatically organizes RAW photography files (`.arw`) into "buckets" based on their dominant color palettes. It respects `.xmp` sidecar edits by attempting to render them before analysis.

## Features

- **XMP-Aware Rendering**: Uses a tiered pipeline to ensure edits are reflected in the color analysis.
- **Perceptual Color Analysis**: Converts images to CIELAB color space for accurate visual distance calculations.
- **Automated Clustering**: Uses K-Means and Agglomerative Clustering to group images with similar palettes.
- **Sidecar Pairing**: Ensures `.arw` and `.xmp` files always stay together during organization.
- **High Performance**: Utilizes multiprocessing for parallel RAW parsing and analysis.

## Prerequisites

### System Requirements
- **Python 3.10+**
- **Operating System**: Linux, macOS, or Windows.

### Optional Dependencies (for High Fidelity)
To ensure that `.xmp` sidecar edits are fully applied during the analysis, it is highly recommended to install **darktable**. The script will automatically detect `darktable-cli` and use it for rendering.

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

The tool is executed via the command line. Basic usage requires an input directory.

### Basic Command
```bash
python sort_by_palette.py --input /path/to/your/photos
```

### Common Options

| Option | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | (Required) | Path to directory containing `.arw` and `.xmp` files. |
| `--output-dir` | | `./output_buckets` | Directory where sorted buckets will be created. |
| `--colors` | `-k` | `3` | Number of dominant colors to extract per image. |
| `--clusters` | | `None` | Force a specific number of target buckets. |
| `--threshold` | | `50.0` | Distance threshold for clustering (used if `--clusters` is not set). |
| `--copy` | | `False` | Copy files instead of moving them (safer for testing). |

### Examples

**Copy photos into buckets based on automatic thresholding:**
```bash
python sort_by_palette.py -i ./my_raws --copy
```

**Move photos into exactly 5 buckets, analyzing 5 dominant colors per image:**
```bash
python sort_by_palette.py -i ./my_raws -k 5 --clusters 5
```

## How it Works

1. **Rendering**: The script looks for `.arw` files. It attempts to render them using `darktable-cli` (applying XMP edits), then falls back to `rawpy`'s embedded preview, and finally to basic demosaicing.
2. **Downscaling**: Images are resized to 400px on the long edge to optimize processing speed.
3. **Color Extraction**: 
   - Pixels are converted from sRGB to **CIELAB** space.
   - **K-Means clustering** is performed on each image to find the $K$ most dominant colors.
   - A feature vector is created from these centroids, ordered by pixel frequency.
4. **Global Clustering**: The feature vectors of all images are clustered using **Agglomerative Clustering** (or KMeans), grouping images with similar overall palettes.
5. **Organization**: Files are moved/copied into folders named by their cluster ID and a human-readable color family (e.g., `bucket_0_warm_gold`).