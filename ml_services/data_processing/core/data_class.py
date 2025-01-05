from typing import Dict, Optional, Union, List, Tuple, Any

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype


import numpy as np
from scipy import stats
from scipy.stats import zscore
from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns
        

class CustomDataFrame(pd.DataFrame):
    """
    A subclass of pandas.DataFrame with additional functionality for data validation
    and statistical analysis.
    """
    
    @property
    def _constructor(self):
        """
        Constructor property required for pandas extension classes.
        Ensures operations return a CustomDataFrame instance.
        """
        return CustomDataFrame

    def __init__(self, *args, **kwargs):
        """Initialize the CustomDataFrame."""
        super().__init__(*args, **kwargs)

    @classmethod
    def read_csv(cls, *args: Any, **kwargs: Any) -> 'CustomDataFrame':
        """
        Custom method to read a CSV and return a CustomDataFrame.
        
        Parameters
        ----------
        *args : Any
            Positional arguments passed to pd.read_csv
        **kwargs : Any
            Keyword arguments passed to pd.read_csv
            
        Returns
        -------
        CustomDataFrame
            A new CustomDataFrame instance containing the CSV data
        """
        return cls(pd.read_csv(*args, **kwargs))

    def validate(self, name: str = None, display_df: Optional[str] = None) -> Dict[str, 'CustomDataFrame']:
        """
        Validate the dataframe to ensure no null values or duplicates are present.
        The user has the option to view results as a printout or within a ZenML dashboard.

        Parameters
        ----------
        name : str
            Name of the dataset.
            
        display_df : Optional[str], optional
            Option for the user to view validation results (e.g., 'Info', 'Describe', 'Head', 'Sample').

        Returns
        -------
        CustomDataFrame
            A DataFrame summarizing validation results.
        """
        if not name:
            raise ValueError("A valid name must be provided for the dataset.")
        elif not isinstance(name, str):
            raise TypeError("Provide a valid str name for dataset")

        validation_results = {
            "Total Rows": len(self),
            "Total Features": len(self.columns),
            "Null Values": self.isna().sum().sum(),
            "Duplicate Rows": self.duplicated().sum(),
            "Outlier Columns": sum(
                (abs(zscore(self[col])) > 3).sum() > 0
                for col in self.columns if is_numeric_dtype(self[col])
            ),
            "Cardinal Columns": sum(
                self[col].nunique() < 0.05 * len(self)
                for col in self.columns
            ),
            "Categorical Columns": sum(
                is_object_dtype(self[col]) for col in self.columns
            ),
            "Numeric Columns": sum(
                is_numeric_dtype(self[col]) for col in self.columns
            ),
            "Date Columns": sum(
                1 for col in self.columns if 'date' in col
            ),
        }

        # Display dataframe information if requested
        if display_df:
            if display_df == "Info":
                self.info()
            elif display_df == "Describe":
                display(self.describe(include="all"))
            elif display_df == "Head":
                display(self.head(10))
            elif display_df == "Sample":
                display(self.sample(min(10, len(self))))
            else:
                raise ValueError(f"Invalid display option: {display_df}")

        # Convert results to a DataFrame
        results_df = CustomDataFrame([validation_results], index=[name])
        return {name : results_df}

    
    def analyze_statistics(self, name : str,
                           target_col : str = 'target',
                           return_df : bool = True,
                           decimal_places: int = 2                           
                           ) -> Tuple['CustomDataFrame', 'CustomDataFrame']:
        """
        Analyze statistics of numerical and categorical features in the dataset.

        Parameters:
        ----------
        name : str, 
            Name of the dataset (for metadata tracking).
        target_col : str, default='target'
            The target column name for correlation analysis.
        return_df : bool, default=True
            Whether to return results as DataFrames.
        decimal_places : int, default=2
            Number of decimal places to round numerical values.

        Returns:
        -------
        tuple[CustomDataFrame, CustomDataFrame]
            A tuple containing numerical and categorical feature analysis results.
        """
        num_stats, cat_stats = [], []

        numeric_cols = self.select_dtypes(include=['number']).columns
        categorical_cols = self.select_dtypes(include=['object', 'category']).columns

        # Numerical Feature Analysis
        for col in numeric_cols:
            if col == target_col:
                continue

            feature = self[col].dropna()
            if feature.empty:
                continue

            percentiles = np.percentile(feature, [1, 25, 50, 75, 99])
            z_scores = stats.zscore(feature)

            # Append statistics
            num_stats.append({
                'Dataset': name,
                'Feature': col,
                'Count': len(feature),
                'Missing': self[col].isnull().sum(),
                'Min': feature.min(),
                '1st Percentile': percentiles[0],
                '25th Percentile': percentiles[1],
                'Median': percentiles[2],
                'Mean': feature.mean(),
                '75th Percentile': percentiles[3],
                '99th Percentile': percentiles[4],
                'Max': feature.max(),
                'StdDev': feature.std(),
                'Skewness': stats.skew(feature),
                'Skew P-Value': stats.skewtest(feature).pvalue,
                'Kurtosis': stats.kurtosis(feature),
                'Kurtosis P-Value': stats.kurtosistest(feature).pvalue,
                'Shapiro-Wilk Statistic': stats.shapiro(feature).statistic,
                'Left Outliers': (z_scores < -3).sum(),
                'Values Below 0.01 Percentile': sum(feature < percentiles[0]),
                'Right Outliers': (z_scores > 3).sum(),
                'Values Above 0.99 Percentile': sum(feature > percentiles[4]),
                'Target Correlation': feature.corr(self[target_col]) if target_col in self else np.nan
            })

        # Categorical Feature Analysis
        for col in categorical_cols:
            if col == target_col:
                continue

            feature = self[col].dropna()
            if feature.empty:
                continue

            value_counts = feature.value_counts()
            mode = value_counts.index[0]
            mode_target_pct = (
                self.loc[feature == mode, target_col].mean() if target_col in self else np.nan
            )

            # Append statistics
            cat_stats.append({
                'Dataset': name,
                'Feature': col,
                'Count': len(feature),
                'Missing': self[col].isnull().sum(),
                'Unique Values': len(value_counts),
                'Mode Class': mode,
                'Mode Count': value_counts.iloc[0],
                'Mode Target Pct': mode_target_pct
            })

        # Create DataFrames
        num_stats_df = CustomDataFrame(num_stats).round(decimal_places).sort_values(by='Feature')
        cat_stats_df = CustomDataFrame(cat_stats).round(decimal_places).sort_values(by='Feature')

        return num_stats_df, cat_stats_df
        
    
    def visualize(self,
              name: str,
              target_col: Optional[str] = None,
              plot_types: Optional[List[str]] = None,
              max_categories: int = 10,
              figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Generate comprehensive visualizations for the dataframe's features.
        
        Parameters
        ----------
        name : str
            Name of the dataset for plot titles
        target_col : Optional[str]
            Target column for relationship analysis plots
        plot_types : Optional[List[str]]
            List of plot types to generate. Options: ['distribution', 'correlation', 
            'categorical', 'boxplot', 'scatter']. If None, generates all plots.
        max_categories : int
            Maximum number of categories to show in categorical plots
        figsize : Tuple[int, int]
            Base figure size for plots
            
        Returns
        -------
        None
            Displays plots using matplotlib/seaborn
        """
        if plot_types is None:
            plot_types = ['distribution', 'correlation', 'categorical', 'boxplot', 'scatter']
            
        # Get numeric and categorical columns
        numeric_cols = self.select_dtypes(include=['number']).columns
        categorical_cols = self.select_dtypes(include=['object', 'category']).columns
        
        # Set the style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 1. Distribution Plots for Numerical Features
        if 'distribution' in plot_types and len(numeric_cols) > 0:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) - 1) // n_cols + 1
            
            fig, axes = plt.subplots(n_rows, n_cols, 
                                    figsize=(figsize[0], figsize[1] * n_rows))
            fig.suptitle(f'{name} - Feature Distributions', fontsize=16)
            
            for idx, col in enumerate(numeric_cols):
                if col == target_col:
                    continue
                ax = axes[idx // n_cols][idx % n_cols] if n_rows > 1 else axes[idx]
                sns.histplot(data=self, x=col, kde=True, ax=ax)
                ax.set_title(f'{col} Distribution')
            
            plt.tight_layout()
            plt.show()
        
        # 2. Correlation Heatmap
        if 'correlation' in plot_types and len(numeric_cols) > 1:
            plt.figure(figsize=figsize)
            correlation_matrix = self[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    fmt='.2f',
                    square=True)
            
            plt.title(f'{name} - Feature Correlations')
            plt.tight_layout()
            plt.show()
        
        # 3. Categorical Feature Plots
        if 'categorical' in plot_types and len(categorical_cols) > 0:
            for col in categorical_cols:
                value_counts = self[col].value_counts()
                if len(value_counts) > max_categories:
                    value_counts = value_counts[:max_categories]
                    
                plt.figure(figsize=figsize)
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'{name} - {col} Value Counts')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
        
        # 4. Box Plots for Numerical Features
        if 'boxplot' in plot_types and len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(data=self[numeric_cols])
            plt.title(f'{name} - Feature Distributions (Box Plots)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        
        # 5. Scatter Plots with Target (if specified)
        if 'scatter' in plot_types and target_col and target_col in self.columns:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) - 1) // n_cols + 1
            
            fig, axes = plt.subplots(n_rows, n_cols, 
                                    figsize=(figsize[0], figsize[1] * n_rows))
            fig.suptitle(f'{name} - Feature Relationships with {target_col}', 
                        fontsize=16)
            
            for idx, col in enumerate(numeric_cols):
                if col == target_col:
                    continue
                ax = axes[idx // n_cols][idx % n_cols] if n_rows > 1 else axes[idx]
                sns.scatterplot(data=self, x=col, y=target_col, ax=ax)
                ax.set_title(f'{col} vs {target_col}')
            
            plt.tight_layout()
            plt.show()