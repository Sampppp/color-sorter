import csv
import numpy as np
from pathlib import Path
from metrics.visual_report import generate_visual_report

def main():
    csv_path = Path("metrics/results.csv")
    input_dir = "./jpg_storage"
    raw = False
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run test_rig.py first.")
        return

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # 1. Filter eligible and convert types
    eligible = []
    for r in results:
        try:
            # Convert numeric fields
            r['composite_score'] = float(r['composite_score'])
            r['silhouette'] = float(r['silhouette'])
            r['balance_score'] = float(r['balance_score'])
            r['separation_ratio'] = float(r['separation_ratio'])
            
            if r.get('is_eligible') == 'True' and not np.isnan(r['composite_score']):
                eligible.append(r)
        except (ValueError, TypeError):
            continue

    # 2. Sort by composite score
    top_3 = sorted(eligible, key=lambda x: x['composite_score'], reverse=True)[:3]

    if not top_3:
        print("No eligible pipelines found in results.csv")
        return

    # 3. Reconstruct the 'params' dictionary for the report generator
    # The report generator expects a 'params' key. In the CSV, params are flat.
    # We need to identify which keys are parameters.
    # Standard keys are: sorter, composite_score, silhouette, balance_score, 
    # separation_ratio, num_clusters, duration, is_eligible, etc.
    core_keys = {
        'sorter', 'composite_score', 'silhouette', 'balance_score', 
        'separation_ratio', 'num_clusters', 'duration', 'is_eligible', 
        'eligibility_reason', 'images_processed', 'db_index'
    }

    final_top_3 = []
    for p in top_3:
        params = {k: v for k, v in p.items() if k not in core_keys}
        # Try to convert param values to int if possible
        for k, v in params.items():
            try:
                params[k] = int(v)
            except ValueError:
                pass
        
        # Create a copy and add the nested params dict
        p_copy = p.copy()
        p_copy['params'] = params
        final_top_3.append(p_copy)

    print(f"Generating report for: {[p['sorter'] for p in final_top_3]}")
    generate_visual_report(final_top_3, input_dir, raw)
    print("Done! Report saved to metrics/visual_report.html")

if __name__ == "__main__":
    main()
