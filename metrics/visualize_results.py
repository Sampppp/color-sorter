import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from pathlib import Path

@click.command()
@click.option('--input', '-i', 'input_csv', type=click.Path(exists=True), default='results.csv', help='CSV file containing experiment results.')
@click.option('--output-dir', default='./metrics/plots', help='Directory to save generated plots.')
def main(input_csv, output_dir):
    """
    Visualizes the performance of different image sorting methods based on experiment results.
    Generates heatmaps, comparison bar charts, and efficiency curves.
    """
    df = pd.read_csv(input_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {input_csv}...")

    # 1. Method Comparison (Average Silhouette Score)
    plt.figure(figsize=(12, 6))
    method_perf = df.groupby('sorter')['silhouette'].mean().sort_values(ascending=False)
    sns.barplot(x=method_perf.index, y=method_perf.values, hue=method_perf.index, palette='viridis', legend=False)
    plt.title('Average Silhouette Score by Sorting Method')
    plt.ylabel('Silhouette Score (Higher is Better)')
    plt.xlabel('Sorting Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path / 'method_comparison.png')
    print("Saved method_comparison.png")

    # 2. Efficiency Curve (Silhouette vs Duration)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='duration', y='silhouette', hue='sorter', style='sorter', s=100)
    plt.title('Sorting Efficiency: Quality vs. Time')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Silhouette Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / 'efficiency_curve.png')
    print("Saved efficiency_curve.png")

    # 3. Parameter Heatmaps
    # We generate heatmaps for sorters that have at least 2 numeric parameters
    sorters = df['sorter'].unique()
    for sorter in sorters:
        subset = df[df['sorter'] == sorter]
        # Find numeric columns that aren't core metrics AND have actual data for this sorter
        core_metrics = {'sorter', 'silhouette', 'db_index', 'num_clusters', 'duration', 'balance_std', 'images_processed'}
        params = [col for col in subset.columns if col not in core_metrics and not subset[col].isna().all()]
        
        if len(params) >= 2:
            p1, p2 = params[0], params[1]
            # Pivot data for heatmap
            pivot_df = subset.pivot_table(index=p1, columns=p2, values='silhouette')
            
            if pivot_df.empty or pivot_df.isna().all().all():
                print(f"Skipping heatmap for {sorter}: No valid data.")
                continue

            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".3f")
            plt.title(f'Silhouette Score Heatmap: {sorter}\n({p1} vs {p2})')
            plt.xlabel(p2)
            plt.ylabel(p1)
            plt.tight_layout()
            plt.savefig(out_path / f'heatmap_{sorter}.png')
            print(f"Saved heatmap_{sorter}.png")

    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()