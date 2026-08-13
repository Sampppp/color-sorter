import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger("color-sorter")

def generate_visual_report(top_pipelines: List[Dict[str, Any]], input_dir: str, raw: bool, output_path: str = "metrics/visual_report.html"):
    """
    Generates an HTML report showing sample images for the top performing pipelines.
    """
    from framework import ingest_images, JPGLoader, RAWLoader
    
    input_path = Path(input_dir)
    sources, loader = ingest_images(input_path, raw)
    
    html_content = f"""
    <html>
    <head>
        <title>Sorting Pipeline Visual Validation</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .pipeline-section {{ background: white; padding: 20px; margin-bottom: 40px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .cluster-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }}
            .cluster-box {{ border: 1px solid #ddd; padding: 10px; border-radius: 4px; background: #fafafa; }}
            .cluster-title {{ font-weight: bold; margin-bottom: 5px; font-size: 0.9em; }}
            .image-grid {{ display: flex; gap: 5px; }}
            .image-grid img {{ width: 100px; height: 100px; object-fit: cover; border-radius: 2px; }}
            .metrics {{ margin-bottom: 15px; font-size: 0.9em; color: #666; }}
        </style>
    </head>
    <body>
        <h1>Sorting Pipeline Visual Validation</h1>
        <p>Top 3 pipelines ranked by Composite Score (Silhouette &times; Balance &times; Separation).</p>
    """

    for i, pipeline in enumerate(top_pipelines):
        sorter = pipeline['sorter']
        params = pipeline['params']
        comp_score = pipeline['composite_score']
        sil = pipeline['silhouette']
        bal = pipeline['balance_score']
        sep = pipeline['separation_ratio']
        
        html_content += f"""
        <div class="pipeline-section">
            <h2>#{i+1}: {sorter}</h2>
            <div class="metrics">
                Params: {params} | Composite: {comp_score:.4f} | Sil: {sil:.4f} | Bal: {bal:.4f} | Sep: {sep:.4f}
            </div>
        """
        
        from metrics.evaluator import SorterRegistry
        ExtractorClass, ClustererClass = SorterRegistry.get_components(sorter)
        
        extractor_params = {}
        clusterer_params = {}
        if "k_colors" in params: extractor_params["k_colors"] = params["k_colors"]
        if "bins" in params: extractor_params["bins"] = params["bins"]
        if "clusters" in params: clusterer_params["n_clusters"] = params["clusters"]
        if "threshold" in params: clusterer_params["threshold"] = params["threshold"]
        if "min_cluster_size" in params: clusterer_params["min_cluster_size"] = params["min_cluster_size"]
        
        try:
            extractor = ExtractorClass(**extractor_params)
        except TypeError:
            extractor = ExtractorClass()
            
        try:
            clusterer = ClustererClass(**clusterer_params)
        except TypeError:
            clusterer = ClustererClass()
        
        features = []
        for s in sources:
            try:
                img = loader.load(s)
                features.append(extractor.extract(img))
            except:
                continue
        
        import numpy as np
        from sklearn.preprocessing import normalize
        features_array = np.array(features)
        features_array = normalize(features_array, axis=1)
        labels, _ = clusterer.cluster(features_array)
        
        cluster_map = {}
        for idx, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = []
            if len(cluster_map[label]) < 5:
                cluster_map[label].append(sources[idx].path.name)
        
        html_content += '<div class="cluster-grid">'
        for label in sorted(cluster_map.keys()):
            if label == -1: continue
            
            html_content += f'<div class="cluster-box"><div class="cluster-title">Cluster {label}</div><div class="image-grid">'
            for img_name in cluster_map[label]:
                img_path = Path(input_dir) / img_name
                # Use relative path from project root for HTML
                rel_path = img_path.relative_to(Path.cwd()) if img_path.is_absolute() else img_path
                html_content += f'<img src="{rel_path}" />'
            html_content += '</div></div>'
            
        html_content += '</div></div>'

    html_content += "</body></html>"
    
    with open(output_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Visual report generated at {output_path}")
