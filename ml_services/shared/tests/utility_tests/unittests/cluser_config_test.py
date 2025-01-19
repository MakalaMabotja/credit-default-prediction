import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from shared.utils import DiagnosticConfig, ClusteringConfig, cluster_rfm_plot

class TestDiagnosticConfig:
    """Test suite for DiagnosticConfig class"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = DiagnosticConfig()
        assert config.min_clusters == 2
        assert config.max_clusters == 10
        assert config.random_state == 42
        assert config.best_clusters == 6
        assert config.diagnosis_method == None
        
    def test_custom_values(self):
        """Test custom configuration values"""
        config = DiagnosticConfig(
            min_clusters=3,
            max_clusters=8,
            random_state=123,
            best_clusters=4, 
            diagnosis_method=[]
        )
        assert config.min_clusters == 3
        assert config.max_clusters == 8
        assert config.random_state == 123
        assert config.best_clusters == 4
        assert config.diagnosis_method == []

class TestClusteringConfig:
    """Test suite for ClusteringConfig class"""
    
    def test_default_values(self):
        """Test default configuration values"""
        config = ClusteringConfig()
        
        # Test default column mapping
        expected_cols = [
            'frequency', 'monetary', 'pct_repay', 'repay_rate', 'duration',
            'days_since_last_loan', 'lender_portion', 'new', 'refinanced',
            'refinance_amount', 'increased_risk', 'target'
        ]
        assert all(col in config.col_map for col in expected_cols)
        
        # Test default columns to transform
        expected_transform_cols = [
            'recency', 'monetary', 'repay_rate', 'duration',
            'days_since_last_loan', 'lender_portion', 'refinance_amount'
        ]
        assert config.cols_to_transform == expected_transform_cols
        
        # Test scaler initialization
        assert isinstance(config.scaler, StandardScaler)
        
    def test_custom_values(self):
        """Test custom configuration values"""
        custom_col_map = {'col1': 'mean', 'col2': 'sum'}
        custom_cols_to_transform = ['col1', 'col2']
        custom_scaler = MinMaxScaler()
        
        config = ClusteringConfig(
            col_map=custom_col_map,
            cols_to_transform=custom_cols_to_transform,
            scaler=custom_scaler
        )
        
        assert config.col_map == custom_col_map
        assert config.cols_to_transform == custom_cols_to_transform
        assert config.scaler == custom_scaler

class TestClusterRFMPlot:
    """Test suite for cluster_rfm_plot function"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for plotting tests"""
        np.random.seed(42)
        return pd.DataFrame({
            'cluster': np.random.randint(0, 3, 100),
            'recency': np.random.randint(1, 100, 100),
            'frequency': np.random.randint(1, 10, 100),
            'monetary': np.random.uniform(100, 1000, 100),
            'target': np.random.uniform(0, 1, 100)
        })
    
    def test_valid_column_plot(self, sample_data, tmp_path):
        """Test plotting with valid column"""
        plot_path = os.path.join(tmp_path, 'test_plot.png')
        cluster_rfm_plot(
            sample_data,
            rfm_col='monetary',
            show_plot=False,
            save_plot=True,
            plot_dir=plot_path
        )
        assert os.path.exists(plot_path)
    
    def test_invalid_column(self, sample_data):
        """Test plotting with invalid column"""
        with pytest.raises(ValueError, match="Invalid column"):
            cluster_rfm_plot(
                sample_data,
                rfm_col='invalid_column',
                show_plot=False
            )
    
    def test_plot_without_save(self, sample_data):
        """Test plotting without saving"""
        # Should not raise any exceptions
        cluster_rfm_plot(
            sample_data,
            rfm_col='recency',
            show_plot=False,
            save_plot=False
        )
    
    @pytest.mark.parametrize("rfm_col", ['recency', 'frequency', 'monetary'])
    def test_all_rfm_columns(self, sample_data, rfm_col):
        """Test plotting with each valid RFM column"""
        # Should not raise any exceptions
        cluster_rfm_plot(
            sample_data,
            rfm_col=rfm_col,
            show_plot=False,
            save_plot=False
        )
    
    def test_invalid_save_directory(self, sample_data):
        """Test plotting with invalid save directory"""
        # Use an invalid directory path
        invalid_path = '/invalid/directory/path/plot.png'
        
        # Should print error message but not raise exception
        cluster_rfm_plot(
            sample_data,
            rfm_col='monetary',
            show_plot=False,
            save_plot=True,
            plot_dir=invalid_path
        )
