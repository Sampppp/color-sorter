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
import click
from sklearn.cluster import KMeans, AgglomerativeClustering

# --- Constants ---
MAX_ANALYSIS_SIZE = 400  # Max long edge for color analysis
RAW_EXTENSIONS = {".ARW", ".arw", ".CR2", ".cr2", ".NEF", ".nef", ".DNG", ".dng"}
JPG_EXTENSIONS = {".JPG", ".jpg", ".JPEG", ".jpeg"}

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

def setup_logging(verbose: bool):
    """Configures global logging based on verbosity."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract feature vectors for a batch of images. Defaults to sequential extraction."""
        return np.array([self.extract(img) for img in images])

    def get_centroid_name(self, centroid: np.ndarray) -> str:
        """
        Return a human-readable name for the centroid. 
        Can be overridden by specific extractors to provide meaningful bucket names.
        """
        return "feature"

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

# --- Shared Clusterer Implementations ---

class KMeansClusterer(BaseClusterer):
    """Clusters images using K-Means."""
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = model.fit_predict(features)
        return labels, model.cluster_centers_

class AgglomerativeClusterer(BaseClusterer):
    """Clusters images using Agglomerative Clustering."""
    def __init__(self, threshold: float, metric: str = 'euclidean'):
        self.threshold = threshold
        self.metric = metric

    def cluster(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # linkage='ward' only supports euclidean. For other metrics, we use 'average' or 'complete'
        linkage = 'ward' if self.metric == 'euclidean' else 'average'
        model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=self.threshold, 
            metric=self.metric, 
            linkage=linkage
        )
        labels = model.fit_predict(features)
        
        # Calculate centroids manually for AgglomerativeClustering
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            cluster_points = features[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        
        return labels, np.array(centroids)

# --- Orchestration ---

def load_image_worker(args) -> Tuple[ImageSource, Optional[np.ndarray]]:
    """
    Worker function for multiprocessing. 
    args: (image_source, loader_class, loader_params)
    """
    image_source, loader_class, loader_params = args
    try:
        loader = loader_class(**loader_params)
        img = loader.load(image_source)
        return image_source, img
    except Exception as e:
        logger.error(f"Error loading {image_source.path.name}: {e}")
        return image_source, None

class SortingFramework:
    def __init__(self, loader: BaseImageLoader, extractor: BaseFeatureExtractor, clusterer: BaseClusterer):
        self.loader = loader
        self.extractor = extractor
        self.clusterer = clusterer

    def analyze_and_sort(self, sources: List[ImageSource], output_path: Optional[Path], move: bool = False):
        # 1. Parallel Image Loading
        loader_class = self.loader.__class__
        loader_params = getattr(self.loader, '__dict__', {}).copy()

        tasks = []
        for source in sources:
            tasks.append((source, loader_class, loader_params))

        loaded_images = []
        valid_sources = []
        
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(load_image_worker, tasks), total=len(tasks), desc="Loading Images"))

        for source, img in results:
            if img is not None:
                valid_sources.append(source)
                loaded_images.append(img)
            else:
                tqdm.write(f"Warning: Could not load {source.path.name}")

        if not loaded_images:
            logger.error("No images could be loaded.")
            return None, None

        # 2. Batch Feature Extraction (Main Process)
        logger.info(f"Extracting features for {len(loaded_images)} images...")
        features_array = self.extractor.extract_batch(loaded_images)

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
            # Use the extractor to determine a human-readable name for the cluster
            cluster_name = self.extractor.get_centroid_name(centroid)
            bucket_name = f"bucket_{label}_{cluster_name}"
            
            bucket_dir = None
            if output_path:
                bucket_dir = output_path / bucket_name
                bucket_dir.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for i, label_val in enumerate(labels):
                if label_val == label:
                    source = valid_sources[i]
                    
                    if output_path and bucket_dir:
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

# --- CLI Helpers ---

def common_options(default_output_dir: Optional[str] = None):
    """Decorator to add common CLI options to sorter scripts."""
    def decorator(f):
        f = click.option('--input', '-i', 'input_dir', type=click.Path(exists=True), help='Input directory containing images.')(f)
        f = click.option('--output-dir', default=default_output_dir, help='Output directory for sorted buckets.')(f)
        f = click.option('--move', is_flag=True, default=False, help='Remove input files after they have been copied to buckets.')(f)
        f = click.option('--raw', is_flag=True, default=False, help='Process RAW files instead of JPGs.')(f)
        f = click.option('--verbose', '-v', is_flag=True, default=False, help='Enable debug logging.')(f)
        return f
    return decorator

def ingest_images(input_path: Path, raw: bool) -> Tuple[List[ImageSource], BaseImageLoader]:
    """Scans directory for images and returns ImageSource list and appropriate loader."""
    sources = []
    if raw:
        logger.info(f"Scanning for RAW files in {input_path}...")
        raw_files = [f for f in input_path.iterdir() if f.suffix in RAW_EXTENSIONS]
        for f in raw_files:
            xmp = f.with_suffix(".xmp")
            sidecars = [xmp] if xmp.exists() else []
            xmp_upper = f.with_suffix(".XMP")
            if xmp_upper.exists() and xmp_upper not in sidecars:
                sidecars.append(xmp_upper)
            sources.append(ImageSource(path=f, sidecars=sidecars))
        return sources, RAWLoader()
    else:
        logger.info(f"Scanning for JPG files in {input_path}...")
        jpg_files = [f for f in input_path.iterdir() if f.suffix in JPG_EXTENSIONS]
        for f in jpg_files:
            sources.append(ImageSource(path=f))
        return sources, JPGLoader()

def print_summary(summary: Dict[str, int], total_processed: int, title: str = "Sorting Summary"):
    """Prints a formatted summary of the sorting results."""
    logger.info(f"\n--- {title} ---")
    logger.info(f"Total images processed: {total_processed}")
    logger.info(f"Buckets created: {len(summary)}")
    for bucket, count in summary.items():
        logger.info(f"{bucket}: {count} files")
    logger.info("-----------------------")

def run_sorting_pipeline(
    input_dir: Optional[str], 
    output_dir: Optional[str], 
    move: bool, 
    raw: bool, 
    verbose: bool, 
    extractor: BaseFeatureExtractor, 
    clusterer: BaseClusterer, 
    title: str = "Sorting Summary"
):
    """Encapsulates the common sorting workflow."""
    setup_logging(verbose)
    
    if not input_dir:
        input_dir = './raw_storage' if raw else './jpg_storage'
    
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else None
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return

    sources, loader = ingest_images(input_path, raw)
    
    if not sources:
        logger.warning(f"No suitable images found in {input_path}.")
        return

    logger.info(f"Found {len(sources)} images. Starting analysis...")

    framework = SortingFramework(loader, extractor, clusterer)
    
    try:
        valid_sources, summary = framework.analyze_and_sort(sources, output_path, move=move)
        if valid_sources:
            print_summary(summary, len(valid_sources), title)
    except Exception as e:
        logger.error(f"An error occurred during sorting: {e}")