import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click
import numpy as np
from pathlib import Path

@click.command()
@click.option('--input', '-i', 'input_csv', type=click.Path(exists=True), default='metrics/results.csv', help='CSV file containing experiment results.')
@click.option('--output-dir', default='./metrics/plots', help='Directory to save generated plots.')
def main(input_csv, output_dir):
    """
    Visualizes the performance of different image sorting methods based on experiment results.
    Generates heatmaps, comparison bar charts, and efficiency curves using the Composite Score.
    """
    df = pd.read_csv(input_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {input_csv}...")

    # We now include all runs but use a penalty score for ineligible ones
    df_viz = df.copy()
    print(f"Visualizing {len(df_viz)} runs.")

    if df_viz.empty:
        print("No data found to visualize. Exiting.")
        return

    # Calculate Quality Score (Higher is Better)
    # Quality = Max_Rank_Sum - Rank_Sum
    # This inverts the Rank Sum so that the best models (lowest sum) are at the top.
    max_score = df_viz['composite_score'].max()
    df_viz['quality_score'] = max_score - df_viz['composite_score']

    # 1. Method Comparison (Average Quality Score)
    plt.figure(figsize=(12, 6))
    method_perf = df_viz.groupby('sorter')['quality_score'].mean().sort_values(ascending=False)
    sns.barplot(x=method_perf.index, y=method_perf.values, hue=method_perf.index, palette='viridis', legend=False)
    plt.title('Average Sorting Quality by Method (Borda Count)')
    plt.ylabel('Quality Score (Higher is Better)')
    plt.xlabel('Sorting Method')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path / 'method_comparison.png')
    print("Saved method_comparison.png")

    # 2. Efficiency Curve (Quality Score vs Duration)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_viz, x='duration', y='quality_score', hue='sorter', style='sorter', s=100)
    plt.title('Sorting Efficiency: Quality vs. Time')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Quality Score (Higher is Better)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / 'efficiency_curve.png')
    print("Saved efficiency_curve.png")

    # 3. Quality Trade-off (Balance vs Separation)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_viz, x='balance_score', y='separation_ratio', hue='sorter', style='sorter', s=100)
    plt.title('Clustering Quality: Balance vs. Separation')
    plt.xlabel('Balance Score (Evenness)')
    plt.ylabel('Separation Ratio (Inter/Intra)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path / 'quality_tradeoff.png')
    print("Saved quality_tradeoff.png")

    # 4. Parameter Heatmaps
    sorters = df_viz['sorter'].unique()
    for sorter in sorters:
        subset = df_viz[df_viz['sorter'] == sorter]
        # Find numeric columns that aren't core metrics
        core_metrics = {
            'sorter', 'composite_score', 'quality_score', 'silhouette', 'db_index', 
            'num_clusters', 'duration', 'balance_score', 'separation_ratio', 
            'is_eligible', 'eligibility_reason', 'images_processed'
        }
        params = [col for col in subset.columns if col not in core_metrics and not subset[col].isna().all()]
        
        if len(params) >= 2:
            p1, p2 = params[0], params[1]
            pivot_df = subset.pivot_table(index=p1, columns=p2, values='quality_score')
            
            if pivot_df.empty or pivot_df.isna().all().all():
                continue

            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt=".3f")
            plt.title(f'Quality Score Heatmap: {sorter}\n({p1} vs {p2})')
            plt.xlabel(p2)
            plt.ylabel(p1)
            plt.tight_layout()
            plt.savefig(out_path / f'heatmap_{sorter}.png')
            print(f"Saved heatmap_{sorter}.png")

    print(f"\nAll visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
