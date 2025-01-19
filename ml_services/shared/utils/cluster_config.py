import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import os

@dataclass
class DiagnosticConfig:
    """Configuration for clustering diagnostics."""
    min_clusters: int = 2
    max_clusters: int = 10
    random_state: int = 42
    best_clusters: int = 6
    diagnosis_method: Optional[List[str]] = None

@dataclass
class ClusteringConfig:
    """ Cponfiguration for cluster transformations required"""
    col_map: Dict[str, str] = field(default_factory=lambda: {
        'frequency': 'count',
        'monetary': 'mean',
        'pct_repay': 'mean',
        'repay_rate': 'mean',
        'duration': 'mean',
        'days_since_last_loan': 'mean',
        'lender_portion': 'mean',
        'new': 'max',
        'refinanced': 'mean',
        'refinance_amount': 'mean',
        'increased_risk': 'mean',
        'target': 'mean'
        })
    cols_to_transform: List[str] = field(default_factory=lambda: [
        'recency', 'monetary', 'repay_rate', 'duration', 
        'days_since_last_loan', 'lender_portion', 'refinance_amount'
        ])
    scaler: StandardScaler = field(default_factory=StandardScaler)

def cluster_rfm_plot(rfm_df: pd.DataFrame, rfm_col: str, show_plot: bool = False, save_plot: bool = False, plot_dir: Optional[str] = None) -> None: 
    """
    Generate cluster analysis plots for a specified RFM metric.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        DataFrame containing RFM data with 'cluster' and target columns.
    rfm_col : str
        Column to plot. Must be one of ['recency', 'frequency', 'monetary'].
    show_plot : bool, optional
        If True, display the plot. Default is False.
    save_plot : bool, optional
        If True, save the plot to the specified directory. Default is False.
    plot_dir : str, optional
        Directory path to save the plot. Default is '{rfm_col}-cluster plot'.

    Raises
    ------
    ValueError
        If an invalid `rfm_col` is provided.

    Example
    -------
    cluster_rfm_plot(df, rfm_col='recency', show_plot=True, save_plot=True, plot_dir='./plots/')
    """

    if rfm_col not in ['recency', 'frequency', 'monetary']:
        raise ValueError(f"Invalid column '{rfm_col}'. Valid options are: ['recency', 'frequency', 'monetary'].")
    
    plot_label_map = {
        'recency': "Average Days Since Last Loan",
        'frequency': "Average number of Loans",
        'monetary': "Average Loan Value"
    }
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="constrained", facecolor="#F5F5DC")

    fig.suptitle(f'Analysis of {plot_label_map[rfm_col]} per Clusters', fontsize=16)
    fig.supylabel(plot_label_map[rfm_col], fontsize=14)

    sns.barplot(data=rfm_df, hue='cluster', y='monetary', palette='tab10', errorbar=None, ax=axs[0])
    axs[0].set_title(f"Bar Plot of {rfm_col}", fontsize=12)

    sns.boxplot(data=rfm_df, hue='cluster', y='monetary', palette='tab10', ax=axs[1])
    axs[1].set_title(f"Box Plot of {rfm_col} Distributions", fontsize=12)

    sns.scatterplot(data=rfm_df, x='target', y='monetary', hue='cluster', palette='tab10', alpha=0.5, ax=axs[2])
    axs[2].set_title(f"Target vs {rfm_col}", fontsize=12)

    for ax in axs:
        ax.set_ylabel('')

    if show_plot:
        plt.show()
        
    if save_plot:
        try:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(plot_dir), exist_ok=True)
            
            plt.savefig(plot_dir)
            print(f"Plot saved to {plot_dir}")
        
        except (FileNotFoundError, OSError) as e:
            print(f"Error saving plot: {e}")