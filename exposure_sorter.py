import logging
import numpy as np
import cv2
import click
import multiprocessing as mp
from pathlib import Path
from framework import (
    BaseFeatureExtractor, 
    KMeansClusterer,
    MAX_ANALYSIS_SIZE,
    common_options,
    run_sorting_pipeline
)

# --- Constants ---
DEFAULT_CLUSTERS = 4

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

# --- Implementations ---

class ExposureExtractor(BaseFeatureExtractor):
    """Extracts luminance profiling features (mean brightness and standard deviation)."""
    def get_centroid_name(self, centroid: np.ndarray) -> str:
        mean_brightness = centroid[0]
        
        if mean_brightness < 30:
            return "night_lowlight"
        if mean_brightness < 60:
            return "indoor_goldenhour"
        if mean_brightness < 85:
            return "daytime"
        return "overexposed_bright"

    def extract(self, image: np.ndarray) -> np.ndarray:
        # 1. Fast Resize
        h, w = image.shape[:2]
        if max(h, w) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        # 2. Convert to LAB space to isolate Luminance (L channel)
        # image is RGB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        l_channel = lab[:, :, 0]
        
        # 3. Calculate Exposure Profile
        # Mean represents average brightness (exposure)
        # Std represents contrast/dynamic range
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        return np.array([mean_brightness, std_brightness], dtype=np.float32)

# --- CLI Implementation ---
@click.command()
@common_options(default_output_dir='./output_exposure')
@click.option('--clusters', '-c', default=DEFAULT_CLUSTERS, type=int, help='Number of lighting categories (e.g., Day, Golden Hour, Low Light, Night).')
def main(input_dir, output_dir, move, raw, verbose, clusters):
    """Sort images by their lighting conditions and exposure profiling."""
    
    # Strategy Selection
    extractor = ExposureExtractor()
    clusterer = KMeansClusterer(n_clusters=clusters)

    run_sorting_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        move=move,
        raw=raw,
        verbose=verbose,
        extractor=extractor,
        clusterer=clusterer,
        title="Exposure Sorting Summary"
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()