import logging
import numpy as np
import cv2
import click
import multiprocessing as mp
from pathlib import Path
from framework import (
    BaseFeatureExtractor, 
    AgglomerativeClusterer,
    MAX_ANALYSIS_SIZE,
    common_options,
    run_sorting_pipeline
)

# --- Constants ---
DEFAULT_BINS = 8

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

# --- Implementations ---

class HistogramExtractor(BaseFeatureExtractor):
    """Extracts a normalized color histogram in HSV space as a feature vector."""
    def __init__(self, bins: int = DEFAULT_BINS):
        self.bins = bins

    def get_centroid_name(self, centroid: np.ndarray) -> str:
        return "distribution"

    def extract(self, image: np.ndarray) -> np.ndarray:
        # 1. Fast Resize
        h, w = image.shape[:2]
        if max(h, w) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # 2. Convert to HSV
        # image is RGB, cv2.cvtColor expects BGR or RGB depending on flag
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 3. Compute Histograms for each channel
        # Hue: 0-179, Saturation: 0-255, Value: 0-255
        hist_h = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])
        
        # 4. Concatenate and Normalize
        feature_vector = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        
        # L2 Normalization to make it a unit vector (crucial for Cosine Similarity)
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        return feature_vector.astype(np.float32)

# --- CLI Implementation ---
@click.command()
@common_options(default_output_dir='./output_histograms')
@click.option('--bins', '-b', default=DEFAULT_BINS, type=int, help='Number of bins per HSV channel.')
@click.option('--threshold', type=float, default=0.5, help='Distance threshold for Agglomerative Clustering.')
def main(input_dir, output_dir, move, raw, verbose, bins, threshold):
    """Sort images by their overall color distribution using HSV histograms."""
    
    # Strategy Selection
    extractor = HistogramExtractor(bins=bins)
    
    # Use Cosine similarity for normalized histograms
    clusterer = AgglomerativeClusterer(threshold=threshold, metric='cosine')

    run_sorting_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        move=move,
        raw=raw,
        verbose=verbose,
        extractor=extractor,
        clusterer=clusterer,
        title="Histogram Sorting Summary"
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()