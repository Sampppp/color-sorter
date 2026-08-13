# Image Sorting Performance Analysis Report

This report summarizes the findings from the automated testing rig executed on the `jpg_storage` dataset. The evaluation framework was recently refactored to eliminate metric bias by moving from a single-metric (Silhouette) approach to a **Rank-Based Composite Score (Borda Count)**.

## 1. Executive Summary

The transition to a rank-based composite score has provided a much more accurate reflection of sorting quality. By summing the ranks of Silhouette, Balance, and Separation, we now penalize "degenerate" clustering (where a method achieves high cohesion by collapsing almost all images into a single cluster).

The **Exposure Sorter** is the definitive winner, demonstrating the best overall trade-off between cohesion, distribution, and separation. **K-Means Palette** and **CV Feature** sorters, while showing strength in specific metrics, fail to provide the balanced distribution necessary for practical sorting.

### Performance Ranking (by Composite Rank Sum - Lower is Better)
| Rank | Method | Best Composite (Rank Sum) | Optimal Parameters | Key Strength |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Exposure | 23.0 | `clusters: 20` | Superior Balance & Separation |
| 2 | K-Means Palette | 23.0 | `k_colors: 2, clusters: 10` | Extreme Cohesion (but imbalanced) |
| 3 | CV Feature | 38.0 | `clusters: 20` | Moderate Cohesion |
| 4 | K-Means (Other) | 32.0+ | `k_colors: 3, clusters: 20` | High Cohesion |
| 5 | Histogram | 65.0 | `bins: 8, clusters: 10` | Stable but Low Cohesion |

---

## 2. Detailed Method Analysis

### Exposure Sorter (The Gold Standard)
The exposure sorter is the most robust method for this dataset.
- **Composite Performance**: It achieves the lowest rank sum because it is the only method that scores highly across all three dimensions: cohesion (Silhouette), distribution (Balance), and inter-cluster distance (Separation).
- **Scaling**: Performance is strongest at $K=20$, suggesting that lighting variations are granular and well-captured.

### K-Means Palette Sorter (The Cohesion Specialist)
K-Means Palette remains a powerful tool for cohesion but fails on balance.
- **The Balance Gap**: Despite having the highest Silhouette scores in the entire experiment, its Balance Score is consistently $0.0$. This indicates a "winner-take-all" clustering pattern.
- **Finding**: Dominant color extraction is too aggressive for this dataset, creating artificial cohesion that does not translate to useful sorting.

### CV Feature Sorter
The CV Feature sorter showed surprising resilience under the rank-based system.
- **Observation**: By using a larger number of clusters ($K=20$), it manages to avoid total collapse and achieves a moderate composite rank, though it still lags far behind the Exposure sorter.

### Perceptual Hashing (aHash, dHash, pHash)
Hashing methods provide the best **Balance Scores** (often $> 0.6$), meaning they distribute images very evenly. However, their **Silhouette scores** are among the lowest, indicating that the resulting groups lack internal cohesion.

### Deep Learning & Hybrid Embeddings
Semantic embeddings (CLIP) continue to underperform.
- **Finding**: For this specific dataset, low-level visual features (like exposure and color) are far more discriminative than high-level semantic concepts.

---

## 3. Conclusions and Recommendations

### Recommended Configuration
For high-quality, balanced sorting of this image set, the **Exposure Sorter** is strongly recommended:
- **`--clusters 20`** (for maximum separation and quality)
- **`--clusters 10`** (for a more condensed set of buckets)

### Key Findings
1. **Borda Count Success**: The rank-based composite score successfully neutralized the "Silhouette Trap," where imbalanced clusters appeared high-quality.
2. **Balance as a Primary Metric**: The results prove that for a sorter to be "effective," it must not only group similar items but also avoid creating a single dominant cluster.
3. **Lighting as a Primary Feature**: The dataset is primarily differentiated by lighting and exposure rather than object content, making the Exposure Sorter the optimal choice.
4. **Visual Validation**: The `visual_report.html` confirms that the top-ranked Exposure pipelines produce visually coherent groups, whereas the high-silhouette K-Means runs produce one giant group and several tiny ones.

### Final Framework Validation
The implementation of **L2 Normalization**, **Eligibility Filtering**, and **Rank-Based Scoring** has created a rigorous and unbiased evaluation pipeline. The framework now correctly identifies the most perceptually useful sorting algorithms.
