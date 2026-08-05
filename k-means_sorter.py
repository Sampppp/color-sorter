import logging
import numpy as np
import cv2
import click
import multiprocessing as mp
from pathlib import Path
from framework import (
    BaseFeatureExtractor, 
    BaseClusterer, 
    KMeansClusterer,
    AgglomerativeClusterer,
    MAX_ANALYSIS_SIZE,
    common_options,
    run_sorting_pipeline,
    get_color_name
)

# --- Constants ---
DEFAULT_K_COLORS = 3

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

# --- Implementations ---

class DominantColorExtractor(BaseFeatureExtractor):
    """Extracts the top K dominant colors in Lab space as a feature vector."""
    def __init__(self, k_colors: int = DEFAULT_K_COLORS):
        self.k_colors = k_colors

    def get_centroid_name(self, centroid: np.ndarray) -> str:
        # Centroid is flattened dominant colors. Use the first one for naming.
        primary_color = centroid[:3]
        return get_color_name(primary_color)

    def extract(self, image: np.ndarray) -> np.ndarray:
        # 1. Fast Resize: Only resize if larger than analysis size
        h, w = image.shape[:2]
        if max(h, w) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # 2. Sample-First: Pick pixels in RGB space before expensive color conversion
        pixels_rgb = image.reshape((-1, 3))
        num_pixels = pixels_rgb.shape[0]
        
        # Reduced sample size from 10,000 to 2,000 for significant speedup
        sample_size = 8000
        if num_pixels > sample_size:
            # Faster sampling using random integers instead of np.random.choice
            idx = np.random.randint(0, num_pixels, sample_size)
            sampled_pixels = pixels_rgb[idx].astype(np.float32)
        else:
            sampled_pixels = pixels_rgb.astype(np.float32)
        
        # 3. Targeted Conversion: Convert only the sampled pixels to Lab space
        # Reshape to (1, N, 3) to satisfy cv2.cvtColor requirements
        sampled_pixels = sampled_pixels.reshape((1, -1, 3))
        pixels_lab = cv2.cvtColor(sampled_pixels, cv2.COLOR_RGB2Lab)
        pixels_lab = pixels_lab.reshape((-1, 3))
        
        # 4. Extract dominant colors using cv2.kmeans for native C++ speed
        # Criteria: Stop if 10 iterations are reached or epsilon is 1.0
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # cv2.kmeans returns (compactness, labels, centers)
        # We use KMEANS_RANDOM_CENTERS for maximum speed
        _, labels, centers = cv2.kmeans(
            pixels_lab, 
            self.k_colors, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Order centroids by frequency
        labels = labels.flatten()
        # Use minlength to ensure counts array is always of size k_colors, 
        # preventing inhomogeneous feature vectors if fewer clusters are found.
        counts = np.bincount(labels, minlength=self.k_colors)
        sorted_indices = np.argsort(counts)[::-1]
        
        # Ensure we only take up to k_colors
        sorted_indices = sorted_indices[:self.k_colors]
        dominant_colors = centers[sorted_indices]
        
        return dominant_colors.flatten()


# --- CLI Implementation ---
@click.command()
@common_options()
@click.option('--colors', '-k', default=DEFAULT_K_COLORS, type=int, help='Number of dominant colors per image.')
@click.option('--clusters', type=int, help='Number of target buckets. If not provided, AgglomerativeClustering with threshold is used.')
@click.option('--threshold', type=float, default=50.0, help='Distance threshold for clustering if --clusters is not provided.')
def main(input_dir, output_dir, move, raw, verbose, colors, clusters, threshold):
    """Sort images by their dominant color palettes."""
    
    # Strategy Selection
    extractor = DominantColorExtractor(k_colors=colors)
    
    if clusters:
        clusterer = KMeansClusterer(n_clusters=clusters)
    else:
        clusterer = AgglomerativeClusterer(threshold=threshold)

    run_sorting_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        move=move,
        raw=raw,
        verbose=verbose,
        extractor=extractor,
        clusterer=clusterer,
        title="Sorting Summary"
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()