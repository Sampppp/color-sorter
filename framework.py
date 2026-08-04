import os
import shutil
import subprocess
import tempfile
import logging
import numpy as np
import cv2
import rawpy
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# --- Constants ---
MAX_ANALYSIS_SIZE = 400  # Max long edge for color analysis

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

# --- Data Models ---

@dataclass
class ImageSource:
    """Represents an image and its associated sidecar files."""
    path: Path
    sidecars: List[Path] = field(default_factory=list)

# --- Color Utility ---
def lab_to_rgb(lab):
    """Convert a single Lab color to RGB hex string."""
    L, a, b = lab
    
    if L < 30: return "dark", "#000000"
    if L > 85: return "bright", "#FFFFFF"
    
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
        if cls.has_darktable():
            logger.debug(f"Attempting darktable-cli render for {raw_path.name}")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # If xmp_path is provided, darktable-cli uses it if it's in the same dir or specified
                # darktable-cli usually looks for .xmp automatically if it matches the filename
                cmd = ["darktable-cli", str(raw_path), tmp_path]
                subprocess.run(cmd, check=True, capture_output=True)
                img = cv2.imread(tmp_path)
                os.unlink(tmp_path)
                if img is not None:
                    logger.debug(f"Successfully rendered {raw_path.name} using darktable-cli")
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.debug(f"darktable-cli failed for {raw_path.name}: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        logger.debug(f"Attempting rawpy embedded preview for {raw_path.name}")
        try:
            with rawpy.imread(str(raw_path)) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        nparr = np.frombuffer(thumb.data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            logger.debug(f"Successfully rendered {raw_path.name} using rawpy thumbnail")
                            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    logger.debug(f"rawpy thumbnail extraction failed for {raw_path.name}: {e}")
                
                logger.debug(f"Attempting rawpy basic demosaicing for {raw_path.name}")
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, bright=1.0)
                logger.debug(f"Successfully rendered {raw_path.name} using rawpy postprocess")
                return rgb
        except Exception as e:
            logger.error(f"Failed to render {raw_path}: {e}")
            raise RuntimeError(f"Failed to render {raw_path}: {e}")

# --- Image Loaders ---

class BaseImageLoader(ABC):
    """Abstract base class for loading images into numpy arrays."""
    @abstractmethod
    def load(self, source: ImageSource) -> np.ndarray:
        pass

class JPGLoader(BaseImageLoader):
    """Loader for JPG/JPEG images."""
    def load(self, source: ImageSource) -> np.ndarray:
        img = cv2.imread(str(source.path))
        if img is None:
            raise RuntimeError(f"Could not read JPG file: {source.path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class RAWLoader(BaseImageLoader):
    """Loader for RAW images using RAWRenderer."""
    def load(self, source: ImageSource) -> np.ndarray:
        xmp_path = None
        for sidecar in source.sidecars:
            if sidecar.suffix.lower() == ".xmp":
                xmp_path = sidecar
                break
        return RAWRenderer.render(source.path, xmp_path)

# --- Modular Framework Interfaces ---

class BaseFeatureExtractor(ABC):
    """Abstract base class for extracting features from a rendered image."""
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract a feature vector from the image."""
        pass

class BaseClusterer(ABC):
    """Abstract base class for clustering feature vectors into buckets."""
    @abstractmethod
    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clusters features and returns (labels, centroids).
        labels: array of cluster indices for each input feature.
        centroids: array of representative feature vectors for each cluster.
        """
        pass

# --- Orchestration ---

def process_single_image_worker(args) -> Tuple[ImageSource, Optional[np.ndarray]]:
    """
    Worker function for multiprocessing. 
    args: (image_source, loader_class, loader_params, extractor_class, extractor_params)
    """
    image_source, loader_class, loader_params, extractor_class, extractor_params = args
    try:
        # Instantiate loader and extractor inside worker to avoid pickling issues
        loader = loader_class(**loader_params)
        extractor = extractor_class(**extractor_params)
        
        img = loader.load(image_source)
        feature_vector = extractor.extract(img)
        
        return image_source, feature_vector
    except Exception as e:
        logger.error(f"Error processing {image_source.path.name}: {e}")
        return image_source, None

class SortingFramework:
    def __init__(self, loader: BaseImageLoader, extractor: BaseFeatureExtractor, clusterer: BaseClusterer):
        self.loader = loader
        self.extractor = extractor
        self.clusterer = clusterer

    def analyze_and_sort(self, sources: List[ImageSource], output_path: Path, move: bool = False):
        # 1. Parallel Feature Extraction
        loader_class = self.loader.__class__
        extractor_class = self.extractor.__class__
        
        # Extract parameters from the instances
        loader_params = getattr(self.loader, '__dict__', {}).copy()
        extractor_params = getattr(self.extractor, '__dict__', {}).copy()

        tasks = []
        for source in sources:
            tasks.append((source, loader_class, loader_params, extractor_class, extractor_params))

        features = []
        valid_sources = []
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_single_image_worker, tasks), total=len(tasks), desc="Analyzing Palettes"))

        for source, feat in results:
            if feat is not None:
                valid_sources.append(source)
                features.append(feat)
            else:
                tqdm.write(f"Warning: Could not process {source.path.name}")

        if not features:
            logger.error("No images could be analyzed.")
            return None, None

        features_array = np.array(features)

        # 2. Global Clustering
        logger.info("Clustering images into buckets...")
        labels, centroids = self.clusterer.cluster(features_array)

        # 3. Organize Files
        unique_labels = np.unique(labels)
        summary = {}
        copied_files = []

        logger.info(f"Organizing {len(valid_sources)} images into {len(unique_labels)} buckets...")
        for label in unique_labels:
            centroid = centroids[label]
            primary_color = centroid[:3]
            color_name = get_color_name(primary_color)
            bucket_name = f"bucket_{label}_{color_name}"
            bucket_dir = output_path / bucket_name
            bucket_dir.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for i, label_val in enumerate(labels):
                if label_val == label:
                    source = valid_sources[i]
                    
                    # Copy primary file
                    dest_primary = bucket_dir / source.path.name
                    shutil.copy2(source.path, dest_primary)
                    copied_files.append(source.path)
                    
                    # Copy sidecars
                    for sidecar in source.sidecars:
                        dest_sidecar = bucket_dir / sidecar.name
                        shutil.copy2(sidecar, dest_sidecar)
                        copied_files.append(sidecar)
                    
                    count += 1
            
            summary[bucket_name] = count

        # 4. Cleanup
        if move:
            logger.info("Cleaning up input directory...")
            for f in tqdm(copied_files, desc="Removing source files"):
                try:
                    f.unlink()
                except Exception as e:
                    tqdm.write(f"Warning: Could not delete {f}: {e}")

        return valid_sources, summary