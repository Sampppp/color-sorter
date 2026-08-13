import csv
import click
import itertools
import sys
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Add project root to sys.path to allow imports from the root directory
sys.path.append(str(Path(__file__).parent.parent))

from metrics.evaluator import evaluate_sorter

# --- Experiment Configuration ---
# Define the parameter sweeps for each sorter
EXPERIMENTS = {
    "k-means": {
        "k_colors": [2, 3, 5],
        "clusters": [5, 10, 20]
    },
    "cv_feature": {
        "clusters": [5, 10, 20]
    },
    "exposure": {
        "clusters": [5, 10, 20]
    },
    "histogram": {
        "bins": [8, 16],
        "clusters": [5, 10, 20]
    },
    "histogram_3d": {
        "bins": [8, 16],
        "clusters": [5, 10, 20]
    },
    "hash_ahash": {
        "clusters": [5, 10, 20]
    },
    "hash_dhash": {
        "clusters": [5, 10, 20]
    },
    "hash_phash": {
        "clusters": [5, 10, 20]
    },
    "exp_baseline": {
        "config": ["baseline"],
        "metric": ["cosine"],
        "clusters": [5, 10, 20]
    },
    "exp_perceptual": {
        "config": ["perceptual"],
        "metric": ["chi2"],
        "clusters": [5, 10, 20]
    },
    "exp_spatial": {
        "config": ["spatial"],
        "metric": ["jsd"],
        "clusters": [5, 10, 20]
    },
    "deep_embeddings": {
        "metric": ["cosine"],
        "clusters": [5, 10, 20]
    },
    "hybrid_embeddings": {
        "metric": ["cosine"],
        "clusters": [5, 10, 20]
    }
}

def generate_param_combinations(params_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Creates a list of all possible parameter combinations from a dictionary of lists."""
    keys = params_dict.keys()
    values = params_dict.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

@click.command()
@click.option('--input', '-i', 'input_dir', type=click.Path(exists=True), default='./jpg_storage', help='Input directory containing images.')
@click.option('--output-csv', default='results.csv', help='CSV file to save results.')
@click.option('--raw', is_flag=True, default=False, help='Process RAW files instead of JPGs.')
@click.option('--sorters', help='Comma-separated list of sorters to test. If omitted, all are tested.')
def main(input_dir, output_csv, raw, sorters):
    """
    Automated testing rig for image sorting methods.
    Performs parameter sweeps and saves metrics to a CSV file.
    """
    target_sorters = sorters.split(',') if sorters else list(EXPERIMENTS.keys())
    
    results_list = []
    
    for sorter_name in target_sorters:
        if sorter_name not in EXPERIMENTS:
            print(f"Warning: Sorter '{sorter_name}' not found in experiment config. Skipping.")
            continue
            
        print(f"\n>>> Testing Sorter: {sorter_name}")
        param_grid = EXPERIMENTS[sorter_name]
        combinations = generate_param_combinations(param_grid)
        
        print(f"Running {len(combinations)} combinations...")
        
        for i, params in enumerate(combinations, 1):
            print(f"  [{i}/{len(combinations)}] Params: {params}", end=" ", flush=True)
            try:
                result = evaluate_sorter(sorter_name, params, input_dir, raw)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    # Flatten result for CSV
                    flat_result = {
                        "sorter": result["sorter"],
                        "num_clusters": result["num_clusters"],
                        "silhouette": result["silhouette"],
                        "db_index": result["db_index"],
                        "balance_score": result["balance_score"],
                        "separation_ratio": result["separation_ratio"],
                        "composite_score": result["composite_score"],
                        "is_eligible": result["is_eligible"],
                        "duration": result["duration"],
                        "images_processed": result["images_processed"]
                    }
                    # Add parameters to the flat result
                    flat_result.update(params)
                    results_list.append(flat_result)
                    print(f"Done")
            except Exception as e:
                print(f"Failed: {e}")

    if not results_list:
        print("No results collected. Exiting.")
        return

    # --- Rank-Based Composite Score (Borda Count) ---
    print("\nCalculating Rank-Based Composite Scores...")
    df = pd.DataFrame(results_list)
    
    # We only rank eligible runs. Ineligible runs get a penalty score.
    eligible_mask = df['is_eligible'] == True
    df_eligible = df[eligible_mask].copy()
    
    if not df_eligible.empty:
        # Metrics to rank (Higher is better)
        metrics_to_rank = ['silhouette', 'balance_score', 'separation_ratio']
        
        # Initialize rank sum
        df_eligible['rank_sum'] = 0
        
        for metric in metrics_to_rank:
            # Rank: 1 is best (highest value), N is worst (lowest value)
            # We use ascending=False because higher raw values must get lower rank numbers.
            ranks = df_eligible[metric].rank(ascending=False, method='min')
            
            # Handle NaNs: assign them the worst possible rank (N + 1)
            nan_mask = ranks.isna()
            ranks[nan_mask] = len(df_eligible) + 1
            
            df_eligible['rank_sum'] += ranks
        
        # Update the original results_list with the new composite_score (rank_sum)
        # Lower rank_sum is better.
        df.loc[eligible_mask, 'composite_score'] = df_eligible['rank_sum']
        
        # Assign a penalty score to ineligible runs so they appear in visualizations
        # Penalty is the worst possible rank sum + 1
        max_rank_sum = 3 * (len(df_eligible) + 1)
        df.loc[~eligible_mask, 'composite_score'] = max_rank_sum
    else:
        print("No eligible runs found for ranking.")

    # Convert back to list of dicts for CSV writing and report generation
    results_list = df.to_dict('records')

    # Write to CSV
    fieldnames = set()
    for r in results_list:
        fieldnames.update(r.keys())
    
    sorted_fieldnames = sorted(list(fieldnames))
    core_metrics = ["sorter", "composite_score", "silhouette", "balance_score", "separation_ratio", "num_clusters", "duration"]
    header = [f for f in core_metrics if f in sorted_fieldnames] + [f for f in sorted_fieldnames if f not in core_metrics]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(results_list)

    # --- Visual Report Generation (Top 3 Distinct Sorters) ---
    eligible_results = [r for r in results_list if r.get('is_eligible', False)]
    
    if eligible_results:
        print("\nSelecting top 3 distinct sorters for visual report...")
        
        # 1. Group by sorter and find the best config (lowest composite_score)
        best_per_sorter = {}
        for r in eligible_results:
            s_name = r['sorter']
            score = r['composite_score']
            if s_name not in best_per_sorter or score < best_per_sorter[s_name]['composite_score']:
                best_per_sorter[s_name] = r
        
        # 2. Sort the distinct sorters by their best score
        sorted_sorters = sorted(best_per_sorter.values(), key=lambda x: x['composite_score'])
        top_3_distinct = sorted_sorters[:3]
        
        if top_3_distinct:
            print(f"Generating visual report for: {[r['sorter'] for r in top_3_distinct]}")
            from metrics.visual_report import generate_visual_report
            
            processed_top_3 = []
            for r in top_3_distinct:
                # Separate core metrics from params
                core_keys = {"sorter", "composite_score", "silhouette", "balance_score", "separation_ratio", "num_clusters", "duration", "is_eligible", "images_processed", "db_index"}
                params = {k: v for k, v in r.items() if k not in core_keys}
                
                processed_top_3.append({
                    'sorter': r['sorter'],
                    'params': params,
                    'composite_score': r['composite_score'],
                    'silhouette': r['silhouette'],
                    'balance_score': r['balance_score'],
                    'separation_ratio': r['separation_ratio']
                })
                
            generate_visual_report(processed_top_3, input_dir, raw)
        else:
            print("\nNo suitable pipelines found for visual report.")
    else:
        print("\nNo eligible pipelines found for visual report.")

    print(f"\nAll experiments complete. Results saved to {output_csv}")

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
