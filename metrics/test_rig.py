import csv
import click
import itertools
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path to allow imports from the root directory
sys.path.append(str(Path(__file__).parent.parent))

from metrics.evaluator import evaluate_sorter

# --- Experiment Configuration ---
# Define the parameter sweeps for each sorter
EXPERIMENTS = {
    "k-means": {
        "k_colors": [2, 3, 5],
        "clusters": [5, 10, 15]
    },
    "cv_feature": {
        "clusters": [5, 10, 15]
    },
    "exposure": {
        "clusters": [4, 8, 12]
    },
    "histogram": {
        "bins": [8, 16],
        "threshold": [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    "histogram_3d": {
        "bins": [8, 16],
        "threshold": [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    "hash_ahash": {
        "threshold": [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    "hash_dhash": {
        "threshold": [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    "hash_phash": {
        "threshold": [0.1, 0.3, 0.5, 0.7, 0.9]
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
                        "balance_std": result["balance_std"],
                        "duration": result["duration"],
                        "images_processed": result["images_processed"]
                    }
                    # Add parameters to the flat result
                    flat_result.update(params)
                    results_list.append(flat_result)
                    print(f"Done (Sil: {result['silhouette']:.3f})")
            except Exception as e:
                print(f"Failed: {e}")

    # Write to CSV
    if not results_list:
        print("No results collected. Exiting.")
        return

    # Get all possible keys for the CSV header
    fieldnames = set()
    for r in results_list:
        fieldnames.update(r.keys())
    
    # Ensure a consistent order for the header
    sorted_fieldnames = sorted(list(fieldnames))
    # Move core metrics to the front
    core_metrics = ["sorter", "silhouette", "db_index", "num_clusters", "duration"]
    header = [f for f in core_metrics if f in sorted_fieldnames] + [f for f in sorted_fieldnames if f not in core_metrics]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"\nAll experiments complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()