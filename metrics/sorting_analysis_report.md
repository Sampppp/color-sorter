# Image Sorting Performance Analysis Report

This report summarizes the findings from the automated testing rig executed on the `jpg_storage` dataset. The goal was to evaluate different image sorting methods and their parameter sensitivity using the Silhouette Coefficient as the primary metric for clustering quality.

## 1. Executive Summary

The **K-Means Palette Sorter** is the most effective method for this dataset, demonstrating exceptionally high cluster cohesion and separation. The **CV Feature Sorter** performed the worst, with negative silhouette scores indicating that images are likely assigned to the wrong clusters.

### Performance Ranking (by Silhouette Score)
| Rank | Method | Best Silhouette | Optimal Parameters |
| :--- | :--- | :--- | :--- |
| 1 | K-Means Palette | 0.936 | `k_colors: 2, clusters: 5` |
| 2 | Exposure | 0.422 | `clusters: 4` |
| 3 | aHash | 0.195 | `threshold: 0.5` |
| 4 | Histogram | 0.181 | `bins: 8, threshold: 0.5` |
| 5 | dHash | 0.051 | `threshold: 0.5` |
| 6 | pHash | 0.045 | `threshold: 0.5` |
| 7 | CV Feature | -0.091 | `clusters: 5` |

---

## 2. Detailed Method Analysis

### K-Means Palette Sorter
This method showed the strongest performance. 
- **Parameter Sensitivity**: The number of dominant colors (`k_colors`) has a significant impact. Using only 2 dominant colors resulted in the highest and most stable scores (>0.92). Increasing `k_colors` to 5 decreased the silhouette score to the 0.65-0.76 range.
- **Cluster Count**: For `k_colors=2`, the quality remained high regardless of the number of buckets. For `k_colors=3`, increasing the number of clusters from 5 to 15 significantly improved the score (0.771 $\rightarrow$ 0.917).

### Exposure Sorter
The exposure sorter provided moderate results.
- **Observation**: It is most effective with a small number of clusters (4), which aligns with the logical categories of lighting (e.g., Night, Golden Hour, Day, Overexposed).

### Histogram Sorter
Performance was generally low, suggesting that global color distribution is less discriminative than dominant color palettes for this specific dataset.
- **Observation**: A threshold of 0.5 provided the peak silhouette score. Increasing bins from 8 to 16 generally decreased the clustering quality.

### Perceptual Hashing (aHash, dHash, pHash)
Hashing methods performed poorly, indicating that the images in the dataset are not near-duplicates and do not share similar structural compositions.
- **aHash** was the most successful of the three.
- **Threshold Collapse**: At thresholds of 0.7 and 0.9, most hashing methods returned `nan`, indicating that the clustering collapsed into a single giant bucket or every image became its own bucket.

### CV Feature Sorter
This method failed to produce meaningful clusters on this dataset, as evidenced by negative silhouette scores. The composite feature vector (ORB, HSV, Laplacian, Canny) likely creates a high-dimensional space where Euclidean distance is not an effective measure of similarity.

---

## 3. Conclusions and Recommendations

### Recommended Configuration
For general-purpose sorting of this image set, the **K-Means Palette Sorter** should be used with the following parameters:
- **`--colors 2`**
- **`--clusters 5`** (or higher if more granular color buckets are needed)

### Key Findings
1. **Dominant Color > Distribution**: Dominant color extraction is far more effective for grouping these images than global color histograms.
2. **Simplicity Wins**: Lowering the complexity of the feature vector (e.g., reducing `k_colors` from 5 to 2) improved the clustering quality.
3. **Threshold Sweet Spot**: For distance-based clustering (Histogram/Hash), a threshold of **0.5** is the optimal balance; values above 0.7 lead to clustering failure.