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
DEFAULT_BINS = 8  # 8x8x8 = 512 dimensions

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

# --- Implementations ---

class Histogram3DExtractor(BaseFeatureExtractor):
    """
    Extracts a 3D color histogram in CIELAB space.
    Uses luminance weighting to reduce the influence of extreme shadows and highlights.
    """
    def __init__(self, bins: int = DEFAULT_BINS):
        self.bins = bins

    def get_centroid_name(self, centroid: np.ndarray) -> str:
        return "3d_distribution"

    def extract(self, image: np.ndarray) -> np.ndarray:
        # 1. Fast Resize
        h, w = image.shape[:2]
        if max(h, w) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # 2. Convert to Lab color space
        # image is RGB, cv2.cvtColor expects RGB if we use COLOR_RGB2Lab
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        
        # 3. Luminance Weighting
        # L channel is index 0. Range is typically [0, 255] in OpenCV's 8-bit Lab
        # We want to down-weight pixels that are too dark (< 30) or too bright (> 220)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Create a weight map: 1.0 for mid-tones, tapering off at extremes
        # Simple linear ramp or Gaussian-like weight
        weights = np.ones_like(l_channel)
        weights[l_channel < 30] = np.clip((l_channel[l_channel < 30] / 30.0), 0, 1)
        weights[l_channel > 220] = np.clip((255.0 - l_channel[l_channel > 220]) / 35.0, 0, 1)
        
        # 4. Compute 3D Histogram
        # Reshape image to (N, 3) and weights to (N,)
        pixels = lab.reshape(-1, 3).astype(np.float32)
        pixel_weights = weights.reshape(-1).astype(np.float32)
        
        # Define bins for Lab: L[0, 255], a[0, 255], b[0, 255] (OpenCV Lab mapping)
        bins_range = [[0, 256], [0, 256], [0, 256]]
        
        hist, _ = np.histogramdd(
            pixels, 
            bins=(self.bins, self.bins, self.bins), 
            range=bins_range, 
            weights=pixel_weights
        )
        
        # 5. Flatten and Normalize
        feature_vector = hist.flatten()
        
        # L2 Normalization for Cosine Similarity
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        return feature_vector.astype(np.float32)

# --- CLI Implementation ---
@click.command()
@common_options()
@click.option('--bins', '-b', default=DEFAULT_BINS, type=int, help='Number of bins per Lab channel (total bins = bins^3).')
@click.option('--threshold', type=float, default=0.5, help='Distance threshold for Agglomerative Clustering.')
def main(input_dir, output_dir, move, raw, verbose, bins, threshold):
    """Sort images by their 3D color distribution in Lab space with luminance weighting."""
    
    # Strategy Selection
    extractor = Histogram3DExtractor(bins=bins)
    
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
        title="3D Lab Histogram Sorting Summary"
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()