import logging
import numpy as np
import cv2
import click
import multiprocessing as mp
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from framework import (
    BaseFeatureExtractor, 
    BaseClusterer, 
    common_options,
    run_sorting_pipeline
)

# --- Logging Setup ---
logger = logging.getLogger("hash-sorter")

# --- Implementations ---

class AHashExtractor(BaseFeatureExtractor):
    """Average Hash (aHash): Scales to 8x8, grayscale, compares to average intensity."""
    def extract(self, image: np.ndarray) -> np.ndarray:
        # Resize to 8x8 and convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        
        # Calculate average pixel intensity
        avg = np.mean(resized)
        
        # Create binary hash: 1 if pixel > avg, else 0
        hash_bits = (resized > avg).astype(np.float32).flatten()
        return hash_bits

class DHashExtractor(BaseFeatureExtractor):
    """Difference Hash (dHash): Scales to 9x8, grayscale, compares adjacent pixels."""
    def extract(self, image: np.ndarray) -> np.ndarray:
        # Resize to 9x8 and convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        
        # Compare adjacent pixels (left vs right)
        # resized[:, 0:-1] is the left pixel, resized[:, 1:] is the right pixel
        diff = resized[:, 0:-1] > resized[:, 1:]
        
        # Create binary hash
        hash_bits = diff.astype(np.float32).flatten()
        return hash_bits

class PHashExtractor(BaseFeatureExtractor):
    """Perceptual Hash (pHash): Uses DCT to keep low frequencies."""
    def extract(self, image: np.ndarray) -> np.ndarray:
        # Resize to 32x32 and convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        
        # Convert to float for DCT
        resized_float = np.float32(resized)
        
        # Compute Discrete Cosine Transform (DCT)
        dct = cv2.dct(resized_float)
        
        # Extract the top-left 8x8 low-frequency coefficients
        low_freq = dct[0:8, 0:8]
        
        # Calculate average of coefficients (excluding the DC component at 0,0)
        # We create a copy to avoid modifying the original low_freq
        coeffs = low_freq.copy()
        coeffs[0, 0] = 0 
        avg = np.mean(coeffs)
        
        # Create binary hash: 1 if coefficient > avg, else 0
        hash_bits = (low_freq > avg).astype(np.float32).flatten()
        return hash_bits

class HammingClusterer(BaseClusterer):
    """Clusters images using Hamming Distance via Agglomerative Clustering."""
    def __init__(self, threshold: float):
        self.threshold = threshold

    def cluster(self, features: np.ndarray):
        # AgglomerativeClustering with hamming metric
        # distance_threshold is used when n_clusters=None
        model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.threshold, 
            metric='hamming', 
            linkage='average'
        )
        labels = model.fit_predict(features)
        
        # Calculate centroids manually
        # For binary hashes, the centroid is the mode (most frequent bit) at each position
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_points = features[labels == label]
            # Compute mode along axis 0
            centroid = np.mean(cluster_points, axis=0) >= 0.5
            centroids.append(centroid.astype(np.float32))
        
        return labels, np.array(centroids)

# --- CLI Implementation ---
@click.command()
@click.option('--method', type=click.Choice(['ahash', 'dhash', 'phash'], case_sensitive=False), 
              required=True, help='Hashing method to use.')
@click.option('--threshold', type=float, default=0.1, help='Hamming distance threshold for clustering (0.0 to 1.0).')
def main(input_dir, output_dir, move, raw, verbose, method, threshold):
    """Sort images based on visual structure using perceptual hashing."""
    
    # Strategy Selection
    method = method.lower()
    if method == 'ahash':
        extractor = AHashExtractor()
    elif method == 'dhash':
        extractor = DHashExtractor()
    else: # phash
        extractor = PHashExtractor()
    
    clusterer = HammingClusterer(threshold=threshold)

    run_sorting_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        move=move,
        raw=raw,
        verbose=verbose,
        extractor=extractor,
        clusterer=clusterer,
        title="Hash Sorting Summary"
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()