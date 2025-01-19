from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import logging
import pickle

from .cluster_config import DiagnosticConfig, ClusteringConfig, cluster_rfm_plot


class CustomerClusters:
    """
    Customer segmentation using RFM analysis and K-means clustering.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with customer transaction data
        col_map (Dict[str, str]): Mapping of column aggregations for RFM calculation
        config (ClusterConfig): Configuration parameters
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        col_map: Optional[Dict[str, str]], 
        col_to_transform: Optional[List[str]],  
        diagnostic_params: DiagnosticConfig
    ):
        self._validate_inputs(df, col_map, col_to_transform)
        self.data = df.copy()
        self.col_map = ClusteringConfig.col_map or col_map
        col_to_transform = ClusteringConfig.cols_to_transform or col_to_transform
        self.rfm = None
        self.k_clusters = None
        self.X = None
        self.model = None
        self.scaler = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _validate_inputs(self, df: pd.DataFrame, col_map: Dict[str, str], col_to_transform) -> None:
        """Validate input data and column mapping"""
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty")
        
        required_cols = {'customer_id', 'disbursement_date', 'New_versus_Repeat', 
                        'Total_Amount', 'ID', 'Lender_portion_Funded'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def customer_rfm(self) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer.
        
        Returns:
            pd.DataFrame: DataFrame with RFM metrics
        """
        try:
            df = self.data
            last_day = df['disbursement_date'].max()
            
            # Calculate recency
            recency = df.groupby('customer_id')['disbursement_date'].agg(
                recency=lambda x: (last_day - x.max()).days
            ).reset_index()
            
            # Rename columns for consistency
            column_mapping = {
                'Total_Amount': 'monetary',
                'ID': 'frequency',
                'Lender_portion_Funded': 'lender_portion'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Aggregate customer metrics
            customer_df = df.groupby('customer_id').agg(self.col_map).reset_index()
            self.rfm = customer_df.merge(recency, on='customer_id', how='left')
            
            return self.rfm
            
        except Exception as e:
            self.logger.error(f"Error in RFM calculation: {str(e)}")
            raise

    def rfm_clusters(
        self, 
        cols_to_transform: List[str], 
        best_k: int = 6
    ) -> pd.DataFrame:
        """
        Perform clustering on RFM metrics.
        
        Parameters:
            cols_to_transform (List[str]): Columns to apply log transformation
            best_k (int): Number of clusters
            
        Returns:
            pd.DataFrame: DataFrame with cluster assignments
        """
        self.k_clusters = best_k
        
        if self.rfm is None:
            self.customer_rfm()
            
        try:
            customer_df = self.rfm
            
            # Log transform specified columns
            customer_df[cols_to_transform] = np.log1p(customer_df[cols_to_transform])
            
            # Prepare features for clustering
            feature_cols = customer_df.columns.difference(['customer_id', 'target'])
            X = customer_df[feature_cols].fillna(0)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Perform clustering
            self.model = KMeans(
                n_clusters=self.k_clusters,
                random_state=self.config.random_state
            )
            clusters = self.model.fit_predict(X_scaled)
            customer_df['cluster'] = clusters
            
            self.X = X_scaled
            
            return customer_df
            
        except Exception as e:
            self.logger.error(f"Error in clustering: {str(e)}")
            raise

    def display_diagnostics(
        self, 
        diagnostics: Optional[List[str]] = None,
        save_plots: bool = False
    ) -> Dict[str, Union[List[float], Dict[int, float]]]:
        """
        Display and return clustering diagnostic metrics.
        
        Parameters:
            diagnostics (List[str]): List of diagnostics to display ('silhouette', 'elbow')
            save_plots (bool): Whether to save plots to disk
            
        Returns:
            Dict containing diagnostic metrics
        """
        results = {}
        
        if self.X is None:
            raise ValueError("Must run rfm_clusters() before diagnostics")
            
        if diagnostics is None or diagnostics == ['all']:
            diagnostics = ['silhouette', 'elbow']

        if 'silhouette' in diagnostics:
            silhouette_scores = self._calculate_silhouette_scores()
            results['silhouette_scores'] = silhouette_scores
            self._plot_silhouette(silhouette_scores, save_plots)

        if 'elbow' in diagnostics:
            inertia_scores = self._calculate_elbow_scores()
            results['inertia_scores'] = inertia_scores
            self._plot_elbow(inertia_scores, save_plots)
            
        return results

    def _calculate_silhouette_scores(self) -> Dict[int, float]:
        """Calculate silhouette scores for different k values"""
        scores = {}
        k_values = range(self.config.min_clusters, self.config.max_clusters + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state)
            labels = kmeans.fit_predict(self.X)
            scores[k] = silhouette_score(self.X, labels)
            
        return scores

    def _calculate_elbow_scores(self) -> Dict[int, float]:
        """Calculate inertia scores for different k values"""
        scores = {}
        k_values = range(1, self.config.max_clusters + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.config.random_state)
            kmeans.fit(self.X)
            scores[k] = kmeans.inertia_
            
        return scores

    def _plot_silhouette(self, scores: Dict[int, float], save: bool = False) -> None:
        """Plot silhouette scores"""
        plt.figure(figsize=(10, 6))
        plt.plot(list(scores.keys()), list(scores.values()), marker='o')
        plt.title("Silhouette Method: Optimal Number of Clusters")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        
        if save:
            plt.savefig('silhouette_scores.png')
        plt.show()

    def _plot_elbow(self, scores: Dict[int, float], save: bool = False) -> None:
        """Plot elbow curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(list(scores.keys()), list(scores.values()), marker='o')
        plt.title("Elbow Method For Optimal k")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Inertia")
        plt.grid(True)
        
        if save:
            plt.savefig('elbow_scores.png')
        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No trained model to save")
            
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config
            }, f)

    @classmethod
    def load_model(cls, filepath: str, df: pd.DataFrame, col_map: Dict[str, str]) -> 'CustomerClusters':
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
            
        instance = cls(df, col_map, config=saved_data['config'])
        instance.model = saved_data['model']
        instance.scaler = saved_data['scaler']
        
        return instance