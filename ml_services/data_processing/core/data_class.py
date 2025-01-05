from typing import Dict, Optional, Union, List, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype


import numpy as np
from scipy import stats
from scipy.stats import zscore
from IPython.display import display

class CustomDataFrame(pd.DataFrame):

    def validate(self, name: str, display_df: Optional[str] = None) -> Dict[str, CustomDataFrame] :
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
                           decimals_places: int = 2                           
                           ) -> tuple[CustomDataFrame, CustomDataFrame]:
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
        
    
    def visualize(self):
        pass