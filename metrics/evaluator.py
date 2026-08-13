import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize

import sys
import logging
from pathlib import Path

logger = logging.getLogger('color-sorter')
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
from hash_sorter import AHashExtractor, DHashExtractor, PHashExtractor
from histogram_sorter import HistogramExtractor
from histogram_sorter_3d import Histogram3DExtractor
from histogram_experiment_sorter import HistogramExperimentExtractor, DistanceMatrixClusterer
from deep_feature_sorter import DeepFeatureExtractor, HybridFeatureExtractor

class SorterRegistry:
    """Registry to map sorter names to their respective components."""
    
    # Mapping: name -> (ExtractorClass, DefaultClustererClass)
    REGISTRY = {
        'k-means': (DominantColorExtractor, KMeansClusterer),
        'cv_feature': (CVFeatureExtractor, ScalingKMeansClusterer),
        'exposure': (ExposureExtractor, KMeansClusterer),
        'histogram': (HistogramExtractor, AgglomerativeClusterer),
        'histogram_3d': (Histogram3DExtractor, AgglomerativeClusterer),
        'hash_ahash': (AHashExtractor, KMeansClusterer),
        'hash_dhash': (DHashExtractor, KMeansClusterer),
        'hash_phash': (PHashExtractor, KMeansClusterer),
        'exp_baseline': (HistogramExperimentExtractor, DistanceMatrixClusterer),
        'exp_perceptual': (HistogramExperimentExtractor, DistanceMatrixClusterer),
        'exp_spatial': (HistogramExperimentExtractor, DistanceMatrixClusterer),
        'deep_embeddings': (DeepFeatureExtractor, AgglomerativeClusterer),
        'hybrid_embeddings': (HybridFeatureExtractor, AgglomerativeClusterer),
    }

    @classmethod
    def get_components(cls, name: str):
        if name not in cls.REGISTRY:
            raise ValueError(f'Sorter {name} not found in registry. Available: {list(cls.REGISTRY.keys())}')
        return cls.REGISTRY[name]

def calculate_separation_ratio(features, labels, centroids):
    """
    Calculates the ratio of average inter-cluster distance to average intra-cluster distance.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    intra_dists = []
    for label in unique_labels:
        if label == -1: continue
        cluster_points = features[labels == label]
        centroid = centroids[label] if label < len(centroids) else np.mean(cluster_points, axis=0)
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_dists.extend(dists)
    
    avg_intra = np.mean(intra_dists) if intra_dists else np.nan

    inter_dists = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    
    avg_inter = np.mean(inter_dists) if inter_dists else np.nan

    if avg_intra == 0 or np.isnan(avg_intra) or np.isnan(avg_inter):
        return np.nan
        
    return avg_inter / avg_intra

def evaluate_sorter(sorter_name, params, input_dir, raw):
    start_time = time.time()
    
    try:
        ExtractorClass, ClustererClass = SorterRegistry.get_components(sorter_name)
        
        extractor_params = {}
        if 'k_colors' in params: extractor_params['k_colors'] = params['k_colors']
        if 'bins' in params: extractor_params['bins'] = params['bins']
        
        clusterer_params = {}
        if 'clusters' in params: clusterer_params['n_clusters'] = params['clusters']
        if 'threshold' in params: clusterer_params['threshold'] = params['threshold']
        if 'min_cluster_size' in params: clusterer_params['min_cluster_size'] = params['min_cluster_size']
        
        extractor = ExtractorClass(**extractor_params)
        clusterer = ClustererClass(**clusterer_params)
        
        input_path = Path(input_dir)
        sources, loader = ingest_images(input_path, raw)
        if not sources:
            return {'sorter': sorter_name, 'error': 'No images found'}
            
    except Exception as e:
        return {'sorter': sorter_name, 'error': str(e)}

    try:
        if 'DeepFeatureExtractor' in str(type(extractor)) or 'HybridFeatureExtractor' in str(type(extractor)):
            batch_size = 32
            features = []
            for i in range(0, len(sources), batch_size):
                batch_sources = sources[i : i + batch_size]
                batch_images = []
                for s in batch_sources:
                    try:
                        batch_images.append(loader.load(s))
                    except:
                        continue
                if batch_images:
                    features.extend(extractor.extract_batch(batch_images))
        else:
            features = []
            for s in sources:
                try:
                    img = loader.load(s)
                    features.append(extractor.extract(img))
                except:
                    continue
        
        if not features:
            return {'sorter': sorter_name, 'error': 'No features extracted'}
            
    except Exception as e:
        return {'sorter': sorter_name, 'error': str(e)}

    try:
        features_array = np.array(features)
        features_array = normalize(features_array, axis=1)
        labels, centroids = clusterer.cluster(features_array)
    except Exception as e:
        return {'sorter': sorter_name, 'error': str(e)}

    end_time = time.time()
    duration = end_time - start_time

    unique_labels, counts = np.unique(labels, return_counts=True)
    num_clusters = len(unique_labels)
    total_images = len(features_array)
    
    is_eligible = True
    eligibility_reason = ''
    
    if not (3 <= num_clusters <= 30):
        is_eligible = False
        eligibility_reason = 'Cluster count outside [3, 30]'
    elif any(c > 0.8 * total_images for c in counts):
        is_eligible = False
        eligibility_reason = 'Single cluster dominates > 80%'

    if 1 < num_clusters < total_images:
        sil_score = silhouette_score(features_array, labels)
        db_index = davies_bouldin_score(features_array, labels)
    else:
        sil_score = np.nan
        db_index = np.nan

    mean_size = np.mean(counts) if len(counts) > 0 else 0
    std_size = np.std(counts) if len(counts) > 0 else 0
    balance_score = 1.0 - (std_size / mean_size) if mean_size > 0 else 0.0
    balance_score = np.clip(balance_score, 0, 1)

    sep_ratio = calculate_separation_ratio(features_array, labels, centroids)

    if is_eligible and not np.isnan(sil_score) and not np.isnan(sep_ratio):
        # Composite score is now calculated in the test rig using Borda Count (rank-based)
        composite_score = np.nan
    else:
        composite_score = np.nan

    return {
        'sorter': sorter_name,
        'params': params,
        'num_clusters': num_clusters,
        'silhouette': sil_score,
        'db_index': db_index,
        'balance_score': balance_score,
        'separation_ratio': sep_ratio,
        'composite_score': composite_score,
        'is_eligible': is_eligible,
        'eligibility_reason': eligibility_reason,
        'duration': duration,
        'images_processed': total_images
    }
