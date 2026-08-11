import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score

import sys
from pathlib import Path
# Add project root to sys.path to allow imports from the root directory
sys.path.append(str(Path(__file__).parent.parent))

from framework import (
    SortingFramework, 
    JPGLoader, 
    RAWLoader, 
    ingest_images,
    KMeansClusterer,
    AgglomerativeClusterer
)

# Import Extractors and Clusterers from sorter scripts
from k_means_sorter import DominantColorExtractor
from cv_feature_sorter import CVFeatureExtractor, ScalingKMeansClusterer
from exposure_sorter import ExposureExtractor
from hash_sorter import AHashExtractor, DHashExtractor, PHashExtractor, HammingClusterer
from histogram_sorter import HistogramExtractor
from histogram_sorter_3d import Histogram3DExtractor

class SorterRegistry:
    """Registry to map sorter names to their respective components."""
    
    # Mapping: name -> (ExtractorClass, DefaultClustererClass)
    REGISTRY = {
        "k-means": (DominantColorExtractor, KMeansClusterer),
        "cv_feature": (CVFeatureExtractor, ScalingKMeansClusterer),
        "exposure": (ExposureExtractor, KMeansClusterer),
        "histogram": (HistogramExtractor, AgglomerativeClusterer),
        "histogram_3d": (Histogram3DExtractor, AgglomerativeClusterer),
        "hash_ahash": (AHashExtractor, HammingClusterer),
        "hash_dhash": (DHashExtractor, HammingClusterer),
        "hash_phash": (PHashExtractor, HammingClusterer),
    }

    @classmethod
    def get_components(cls, name: str):
        if name not in cls.REGISTRY:
            raise ValueError(f"Sorter '{name}' not found in registry. Available: {list(cls.REGISTRY.keys())}")
        return cls.REGISTRY[name]

def evaluate_sorter(
    sorter_name: str, 
    params: Dict[str, Any], 
    input_dir: str, 
    raw: bool = False
) -> Dict[str, Any]:
    """
    Evaluates a sorting method and returns performance metrics.
    Runs in 'headless' mode (no file copying).
    """
    # 1. Resolve components from registry
    ExtractorClass, ClustererClass = SorterRegistry.get_components(sorter_name)
    
    # 2. Setup components with provided params
    # Separate params for extractor and clusterer based on common naming conventions
    # This is a heuristic; in a real system we might use a more explicit config
    extractor_params = {}
    clusterer_params = {}
    
    # Heuristics for parameter distribution
    if "k_colors" in params: extractor_params["k_colors"] = params["k_colors"]
    if "bins" in params: extractor_params["bins"] = params["bins"]
    if "clusters" in params: clusterer_params["n_clusters"] = params["clusters"]
    if "threshold" in params: clusterer_params["threshold"] = params["threshold"]
    if "metric" in params: clusterer_params["metric"] = params["metric"]
    
    # Special case for CVFeatureExtractor which might take n_clusters in __init__
    if sorter_name == "cv_feature" and "clusters" in params:
        extractor_params["n_clusters"] = params["clusters"]

    extractor = ExtractorClass(**extractor_params)
    clusterer = ClustererClass(**clusterer_params)
    
    # 3. Ingest images
    input_path = Path(input_dir)
    sources, loader = ingest_images(input_path, raw)
    
    if not sources:
        return {"error": "No images found"}

    # 4. Run analysis and clustering (Timing this part)
    start_time = time.time()
    
    # We use a modified version of the framework logic to avoid file operations
    # We can just use the framework's internal methods if we pass output_path=None
    framework = SortingFramework(loader, extractor, clusterer)
    
    # We only need the features and labels for metrics
    # To avoid modifying framework.py, we can replicate the core logic here or 
    # just call analyze_and_sort with output_path=None
    # Note: analyze_and_sort returns (valid_sources, summary)
    # But we need the labels and features for silhouette/DB scores.
    
    # Re-implementing the core loop for metric extraction
    from framework import process_single_image_worker
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    loader_class = loader.__class__
    extractor_class = extractor.__class__
    loader_params = getattr(loader, '__dict__', {}).copy()
    extractor_params_inst = getattr(extractor, '__dict__', {}).copy()

    tasks = [(s, loader_class, loader_params, extractor_class, extractor_params_inst) for s in sources]
    
    features = []
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_single_image_worker, tasks))

    for _, feat in results:
        if feat is not None:
            features.append(feat)
    
    features_array = np.array(features)
    labels, centroids = clusterer.cluster(features_array)
    
    end_time = time.time()
    duration = end_time - start_time

    # 5. Calculate Metrics
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    # Silhouette and DB Index require at least 2 clusters and < N clusters
    if 1 < num_clusters < len(features_array):
        sil_score = silhouette_score(features_array, labels)
        db_index = davies_bouldin_score(features_array, labels)
    else:
        sil_score = np.nan
        db_index = np.nan

    # Cluster Balance: Std Dev of cluster sizes
    counts = np.bincount(labels)
    balance_std = np.std(counts) if len(counts) > 0 else np.nan

    return {
        "sorter": sorter_name,
        "params": params,
        "num_clusters": num_clusters,
        "silhouette": sil_score,
        "db_index": db_index,
        "balance_std": balance_std,
        "duration": duration,
        "images_processed": len(features)
    }