import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Tuple, Optional, List
from sentence_transformers import SentenceTransformer
from framework import (
    BaseFeatureExtractor, 
    KMeansClusterer,
    AgglomerativeClusterer,
    MAX_ANALYSIS_SIZE,
    common_options,
    run_sorting_pipeline
)

# --- Logging Setup ---
logger = logging.getLogger("color-sorter")

class DeepFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts semantic embeddings using a pre-trained CLIP model.
    Model: clip-ViT-B-32 via sentence-transformers.
    """
    def __init__(self, model_name: str = 'clip-ViT-B-32'):
        self.model_name = model_name
        # Load model once during initialization
        self.device = 'cuda' # if torch.cuda.is_available() else 'cuda'
        logger.info(f"Loading deep learning model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def get_centroid_name(self, centroid: np.ndarray) -> str:
        return "semantic_feature"

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts a 512D embedding from a single image.
        """
        return self.extract_batch([image])[0]

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extracts embeddings for a batch of images.
        """
        processed_images = []
        for img in images:
            h, w = img.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            processed_images.append(img)

        # Generate embeddings in batch
        with torch.no_grad():
            embeddings = self.model.encode(processed_images, batch_size=32, show_progress_bar=False)
        
        # Normalize each embedding to unit vector for Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)
            
        return embeddings.astype(np.float32)

class HybridFeatureExtractor(BaseFeatureExtractor):
    """
    Combines semantic deep embeddings with color histograms.
    Weighting: 80% Deep / 20% Color.
    """
    def __init__(self, model_name: str = 'clip-ViT-B-32', bins: int = 8):
        self.deep_extractor = DeepFeatureExtractor(model_name)
        self.bins = bins

    def get_centroid_name(self, centroid: np.ndarray) -> str:
        return "hybrid_feature"

    def _compute_lab_hist(self, image: np.ndarray) -> np.ndarray:
        """Computes a normalized 1D CIELAB histogram."""
        # Resize for consistency
        h, w = image.shape[:2]
        if max(h, w) > MAX_ANALYSIS_SIZE:
            scale = MAX_ANALYSIS_SIZE / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        hists = []
        for i in range(3):
            hist = cv2.calcHist([img_lab], [i], None, [self.bins], [0, 256])
            hists.append(hist.flatten())
        
        feat = np.concatenate(hists)
        norm = np.sum(feat)
        return feat / (norm + 1e-7)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extracts a hybrid embedding for a single image."""
        return self.extract_batch([image])[0]

    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Extracts hybrid embeddings for a batch of images."""
        # 1. Get Deep Embeddings (Batch)
        deep_feats = self.deep_extractor.extract_batch(images)
        
        # 2. Get Color Histograms (Batch)
        color_feats = []
        for img in images:
            color_feats.append(self._compute_lab_hist(img))
        color_feats = np.array(color_feats)
        
        # L2 normalize color features
        color_norms = np.linalg.norm(color_feats, axis=1, keepdims=True)
        color_feats = np.divide(color_feats, color_norms, out=np.zeros_like(color_feats), where=color_norms != 0)
        
        # 3. Apply Weighting (80/20)
        weighted_deep = deep_feats * 0.8
        weighted_color = color_feats * 0.2
        
        combined = np.concatenate([weighted_deep, weighted_color], axis=1)
        
        # Final normalization
        final_norms = np.linalg.norm(combined, axis=1, keepdims=True)
        combined = np.divide(combined, final_norms, out=np.zeros_like(combined), where=final_norms != 0)
            
        return combined.astype(np.float32)

# --- CLI Implementation ---
import click
import multiprocessing as mp

@click.command()
@common_options()
@click.option('--mode', type=click.Choice(['deep', 'hybrid']), default='deep', help='Feature extraction mode.')
@click.option('--threshold', type=float, default=0.5, help='Distance threshold for Agglomerative Clustering.')
def main(input_dir, output_dir, move, raw, verbose, mode, threshold):
    """Sort images using Deep or Hybrid semantic features."""
    
    if mode == 'deep':
        extractor = DeepFeatureExtractor()
        title = "Deep Embedding Sorting"
    else:
        extractor = HybridFeatureExtractor()
        title = "Hybrid Embedding Sorting"
    
    # Use Cosine similarity for embeddings
    clusterer = AgglomerativeClusterer(threshold=threshold, metric='cosine')
    
    run_sorting_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        move=move,
        raw=raw,
        verbose=verbose,
        extractor=extractor,
        clusterer=clusterer,
        title=title
    )

if __name__ == "__main__":
    # Use 'spawn' for torch/multiprocessing compatibility
    mp.set_start_method('spawn', force=True)
    main()