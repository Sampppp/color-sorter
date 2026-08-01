import os
import shutil
import subprocess
import tempfile
import numpy as np
import cv2
import rawpy
import click
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import KMeans, AgglomerativeClustering
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# --- Constants ---
MAX_ANALYSIS_SIZE = 400  # Max long edge for color analysis
DEFAULT_K_COLORS = 3

# --- Color Utility ---
def lab_to_rgb(lab):
    """Convert a single Lab color to RGB hex string."""
    # This is a simplified approximation for bucket naming
    # For precise naming, we'd use a full color-name library or a lookup table
    # Here we use a basic mapping of Lab ranges to names
    L, a, b = lab
    
    if L < 30: return "dark", "#000000"
    if L > 85: return "bright", "#FFFFFF"
    
    # Basic color family detection based on a and b
    if abs(a) < 5 and abs(b) < 5:
        return "grayscale", "#808080"
    if a > 10 and b > 10:
        return "warm_gold", "#FFD700"
    if a < -10 and b < -10:
        return "cool_blue", "#0000FF"
    if a > 10 and b < -10:
        return "magenta", "#FF00FF"
    if a < -10 and b > 10:
        return "green", "#00FF00"
    
    return "neutral", "#C0C0C0"

def get_color_name(lab_centroid):
    """Returns a human-readable name for a Lab color."""
    return lab_to_rgb(lab_centroid)[0]

# --- Rendering Pipeline ---
class RAWRenderer:
    @staticmethod
    def has_darktable():
        try:
            subprocess.run(["darktable-cli", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @classmethod
    def render(cls, raw_path: Path, xmp_path: Optional[Path] = None) -> np.ndarray:
        """Renders a RAW file to an RGB numpy array."""
        # Strategy 1: darktable-cli (Best for XMP)
        if cls.has_darktable():
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                cmd = ["darktable-cli", str(raw_path), tmp_path]
                # darktable-cli automatically looks for .xmp with the same name in the same dir
                # but we can be explicit if needed.
                subprocess.run(cmd, check=True, capture_output=True)
                img = cv2.imread(tmp_path)
                os.unlink(tmp_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                # Fallback to rawpy if darktable fails
                pass

        # Strategy 2: rawpy embedded preview (Fast)
        try:
            with rawpy.imread(str(raw_path)) as raw:
                try:
                    # Try to get the best embedded thumbnail
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        # thumb.data is the JPEG bytes
                        nparr = np.frombuffer(thumb.data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass
                
                # Strategy 3: rawpy basic demosaicing (Slowest)
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0)
                return rgb
        except Exception as e:
            raise RuntimeError(f"Failed to render {raw_path}: {e}")

# --- Analysis Pipeline ---
def process_single_image(args) -> Tuple[Path, Optional[np.ndarray]]:
    """Worker function for multiprocessing."""
    raw_path, xmp_path, k_colors = args
    try:
        # Render
        img = RAWRenderer.render(raw_path, xmp_path)
        
        # Resize for performance
        h, w = img.shape[:2]
        if max(h, w) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # Convert to Lab
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        
        # Flatten for K-Means
        pixels = img_lab.reshape((-1, 3)).astype(np.float32)
        
        # Extract dominant colors
        kmeans = KMeans(n_clusters=k_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Order centroids by frequency
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = kmeans.cluster_centers_[sorted_indices]
        
        # Create 1D feature vector: [L1, a1, b1, L2, a2, b2, ...]
        feature_vector = dominant_colors.flatten()
        return raw_path, feature_vector
    except Exception as e:
        return raw_path, None

# --- CLI Implementation ---
@click.command()
@click.option('--input', '-i', 'input_dir', required=True, type=click.Path(exists=True), help='Input directory containing .ARW and .XMP files.')
@click.option('--output-dir', default='./output_buckets', help='Output directory for sorted buckets.')
@click.option('--colors', '-k', default=DEFAULT_K_COLORS, type=int, help='Number of dominant colors per image.')
@click.option('--clusters', type=int, help='Number of target buckets. If not provided, AgglomerativeClustering with threshold is used.')
@click.option('--threshold', type=float, default=50.0, help='Distance threshold for clustering if --clusters is not provided.')
@click.option('--copy', is_flag=True, default=False, help='Copy files instead of moving them.')
def main(input_dir, output_dir, colors, clusters, threshold, copy):
    """Sort RAW images by their dominant color palettes."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 1. Collect files
    raw_files = list(input_path.glob("*.ARW")) + list(input_path.glob("*.arw"))
    if not raw_files:
        click.echo("No .ARW files found in the input directory.")
        return

    click.echo(f"Found {len(raw_files)} RAW files. Starting analysis...")

    # Prepare tasks for multiprocessing
    tasks = []
    for raw in raw_files:
        xmp = raw.with_suffix(".xmp") if not raw.suffix.lower() == ".xmp" else None
        if not xmp.exists() if xmp else False:
            xmp = None # Handle missing XMP gracefully
        tasks.append((raw, xmp, colors))

    # 2. Parallel Processing
    features = []
    valid_raws = []
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc="Analyzing Palettes"))

    for raw, feat in results:
        if feat is not None:
            valid_raws.append(raw)
            features.append(feat)
        else:
            click.echo(f"Warning: Could not process {raw.name}")

    if not features:
        click.echo("No images could be analyzed. Exiting.")
        return

    features_array = np.array(features)

    # 3. Global Clustering
    click.echo("Clustering images into buckets...")
    if clusters:
        model = KMeans(n_clusters=clusters, n_init=10, random_state=42)
        labels = model.fit_predict(features_array)
        centroids = model.cluster_centers_
    else:
        # Use AgglomerativeClustering for automatic bucket count based on threshold
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
        labels = model.fit_predict(features_array)
        # Calculate centroids for naming
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_points = features_array[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        centroids = np.array(centroids)

    # 4. Organize Files
    unique_labels = np.unique(labels)
    summary = {}

    for label in unique_labels:
        # Generate bucket name based on the first dominant color of the cluster centroid
        centroid = centroids[label]
        # The centroid is [L1, a1, b1, L2, a2, b2, ...]
        primary_color = centroid[:3]
        color_name = get_color_name(primary_color)
        bucket_name = f"bucket_{label}_{color_name}"
        bucket_dir = output_path / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, label_val in enumerate(labels):
            if label_val == label:
                raw_file = valid_raws[i]
                # Find matching XMP
                xmp_file = raw_file.with_suffix(".xmp")
                if not xmp_file.exists():
                    # Try case insensitive
                    xmp_file_lower = raw_file.with_suffix(".XMP")
                    if xmp_file_lower.exists():
                        xmp_file = xmp_file_lower
                    else:
                        xmp_file = None

                # Move/Copy RAW
                dest_raw = bucket_dir / raw_file.name
                if copy:
                    shutil.copy2(raw_file, dest_raw)
                else:
                    shutil.move(str(raw_file), str(dest_raw))
                
                # Move/Copy XMP
                if xmp_file:
                    dest_xmp = bucket_dir / xmp_file.name
                    if copy:
                        shutil.copy2(xmp_file, dest_xmp)
                    else:
                        shutil.move(str(xmp_file), str(dest_xmp))
                
                count += 1
        
        summary[bucket_name] = count

    # 5. Final Summary
    click.echo("\n--- Sorting Summary ---")
    click.echo(f"Total images processed: {len(valid_raws)}")
    click.echo(f"Buckets created: {len(unique_labels)}")
    for bucket, count in summary.items():
        click.echo(f"{bucket}: {count} files")
    click.echo("-----------------------")

if __name__ == "__main__":
    main()