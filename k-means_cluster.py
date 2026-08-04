import logging
import numpy as np
import cv2
import click
import multiprocessing as mp
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from framework import (
    SortingFramework, 
    BaseFeatureExtractor, 
    BaseClusterer, 
    MAX_ANALYSIS_SIZE,
    ImageSource,
    JPGLoader,
    RAWLoader
)

# --- Constants ---
DEFAULT_K_COLORS = 3
RAW_EXTENSIONS = {".ARW", ".arw", ".CR2", ".cr2", ".NEF", ".nef", ".DNG", ".dng"}
JPG_EXTENSIONS = {".JPG", ".jpg", ".JPEG", ".jpeg"}

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

# --- Implementations ---

class DominantColorExtractor(BaseFeatureExtractor):
    """Extracts the top K dominant colors in Lab space as a feature vector."""
    def __init__(self, k_colors: int = DEFAULT_K_COLORS):
        self.k_colors = k_colors

    def extract(self, image: np.ndarray) -> np.ndarray:
        # Resize for performance
        h, w = image.shape[:2]
        if max(h, w) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # Convert to Lab
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        pixels = img_lab.reshape((-1, 3)).astype(np.float32)
        
        # Extract dominant colors using K-Means
        kmeans = KMeans(n_clusters=self.k_colors, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Order centroids by frequency
        labels = kmeans.labels_
        counts = np.bincount(labels)
        sorted_indices = np.argsort(counts)[::-1]
        dominant_colors = kmeans.cluster_centers_[sorted_indices]
        
        return dominant_colors.flatten()

class KMeansClusterer(BaseClusterer):
    """Clusters images using K-Means."""
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def cluster(self, features: np.ndarray):
        model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = model.fit_predict(features)
        return labels, model.cluster_centers_

class AgglomerativeClusterer(BaseClusterer):
    """Clusters images using Agglomerative Clustering."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def cluster(self, features: np.ndarray):
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=self.threshold)
        labels = model.fit_predict(features)
        
        # Calculate centroids manually for AgglomerativeClustering
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_points = features[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        
        return labels, np.array(centroids)

# --- CLI Implementation ---
@click.command()
@click.option('--input', '-i', 'input_dir', type=click.Path(exists=True), help='Input directory containing images.')
@click.option('--output-dir', default='./output_buckets', help='Output directory for sorted buckets.')
@click.option('--colors', '-k', default=DEFAULT_K_COLORS, type=int, help='Number of dominant colors per image.')
@click.option('--clusters', type=int, help='Number of target buckets. If not provided, AgglomerativeClustering with threshold is used.')
@click.option('--threshold', type=float, default=50.0, help='Distance threshold for clustering if --clusters is not provided.')
@click.option('--move', is_flag=True, default=False, help='Remove input files after they have been copied to buckets.')
@click.option('--raw', is_flag=True, default=False, help='Process RAW files instead of JPGs.')
@click.option('--verbose', '-v', is_flag=True, default=False, help='Enable debug logging.')
def main(input_dir, output_dir, colors, clusters, threshold, move, raw, verbose):
    """Sort images by their dominant color palettes."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Default input directory if not provided
    if not input_dir:
        input_dir = './raw_storage' if raw else './jpg_storage'
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    # Ingestion Logic
    sources = []
    if raw:
        logger.info(f"Scanning for RAW files in {input_path}...")
        raw_files = [f for f in input_path.iterdir() if f.suffix in RAW_EXTENSIONS]
        for f in raw_files:
            xmp = f.with_suffix(".xmp")
            sidecars = [xmp] if xmp.exists() else []
            # Also check for uppercase .XMP
            xmp_upper = f.with_suffix(".XMP")
            if xmp_upper.exists() and xmp_upper not in sidecars:
                sidecars.append(xmp_upper)
            sources.append(ImageSource(path=f, sidecars=sidecars))
        loader = RAWLoader()
    else:
        logger.info(f"Scanning for JPG files in {input_path}...")
        jpg_files = [f for f in input_path.iterdir() if f.suffix in JPG_EXTENSIONS]
        for f in jpg_files:
            sources.append(ImageSource(path=f))
        loader = JPGLoader()

    if not sources:
        logger.warning(f"No suitable images found in {input_path}.")
        return

    logger.info(f"Found {len(sources)} images. Starting analysis...")

    # Strategy Selection
    extractor = DominantColorExtractor(k_colors=colors)
    
    if clusters:
        clusterer = KMeansClusterer(n_clusters=clusters)
    else:
        clusterer = AgglomerativeClusterer(threshold=threshold)

    framework = SortingFramework(loader, extractor, clusterer)
    
    try:
        valid_sources, summary = framework.analyze_and_sort(sources, output_path, move=move)
        
        if valid_sources:
            logger.info("\n--- Sorting Summary ---")
            logger.info(f"Total images processed: {len(valid_sources)}")
            logger.info(f"Buckets created: {len(summary)}")
            for bucket, count in summary.items():
                logger.info(f"{bucket}: {count} files")
            logger.info("-----------------------")
    except Exception as e:
        logger.error(f"An error occurred during sorting: {e}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()