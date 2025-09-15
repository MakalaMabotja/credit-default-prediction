import os 
import sys 

sys.path.append(os.path.abspath(""))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from shared.utils import CustomerClusters, DiagnosticConfig, ClusteringConfig
from shared.core import CustomDataFrame

# Replace with your test file
# TEST_FILE = 'C:/Python Projects/2-CreditScore Predictions/train_processed.csv'
TEST_FILE = 'C:/Python Projects/2-CreditScore Predictions/african-credit-scoring-challenge20241129-14702-1nqro8v/Train.csv'

# Fixture for test data
@pytest.fixture
def test_df():
    """Create a test DataFrame fixture."""
    return CustomDataFrame.read_csv(TEST_FILE)

@pytest.fixture
def sample_data():
    """Create sample transaction data for testing"""
    dates = [
        datetime.now() - timedelta(days=x) 
        for x in [0, 1, 5, 10, 15, 20]
    ]
    
    return pd.DataFrame({
        'customer_id': [1, 1, 2, 2, 3, 3],
        'disbursement_date': dates,
        'New_versus_Repeat': ['New', 'Repeat', 'New', 'Repeat', 'New', 'Repeat'],
        'Total_Amount': [100, 200, 150, 250, 300, 400],
        'ID': [1, 2, 1, 2, 1, 2],
        'Lender_portion_Funded': [0.8, 0.9, 0.7, 0.85, 0.95, 0.88]
    })

@pytest.fixture
def column_mapping():
    """Create sample column mapping for RFM calculation"""
    return {
        'monetary': 'sum',
        'frequency': 'count',
        'lender_portion': 'mean',
        'New_versus_Repeat': 'last'
    }

def test_customercluster_initialization(test_df, column_mapping):
    """Test CustomerClusters initialization"""
    clusters = CustomerClusters(test_df, column_mapping)
    assert clusters.data is not None
    assert clusters.col_map == column_mapping
    assert isinstance(clusters.diagnostic_params, ClusteringConfig)

def test_validation_empty_dataframe(column_mapping):
    """Test initialization with empty DataFrame"""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty"):
        CustomerClusters(empty_df, column_mapping)

def test_validation_missing_columns(column_mapping):
    """Test initialization with missing required columns"""
    invalid_df = pd.DataFrame({
        'customer_id': [1, 2],
        'Total_Amount': [100, 200]
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        CustomerClusters(invalid_df, column_mapping)

def test_customer_rfm_calculation(test_df, column_mapping):
    """Test RFM metrics calculation"""
    clusters = CustomerClusters(test_df, column_mapping)
    rfm_df = clusters.customer_rfm()
    
    assert isinstance(rfm_df, pd.DataFrame)
    assert 'customer_id' in rfm_df.columns
    assert 'recency' in rfm_df.columns
    assert len(rfm_df) == 3  # Unique customers
    
    # Test recency calculation
    assert all(rfm_df['recency'] >= 0)
    
    # Test monetary calculation
    customer_1_total = test_df[test_df['customer_id'] == 1]['Total_Amount'].sum()
    assert rfm_df[rfm_df['customer_id'] == 1]['monetary'].iloc[0] == customer_1_total

def test_rfm_clusters(test_df, column_mapping):
    """Test clustering functionality"""
    clusters = CustomerClusters(test_df, column_mapping)
    cols_to_transform = ['monetary', 'frequency']
    clustered_df = clusters.rfm_clusters(cols_to_transform, best_k=2)
    
    assert isinstance(clustered_df, pd.DataFrame)
    assert 'cluster' in clustered_df.columns
    assert clustered_df['cluster'].nunique() == 2
    assert clusters.model is not None
    assert clusters.scaler is not None

def test_display_diagnostics(test_df, column_mapping):
    """Test diagnostic metrics calculation"""
    clusters = CustomerClusters(test_df, column_mapping)
    cols_to_transform = ['monetary', 'frequency']
    clusters.rfm_clusters(cols_to_transform, best_k=2)
    
    # Test with specific diagnostics
    results = clusters.display_diagnostics(['silhouette'])
    assert 'silhouette_scores' in results
    assert isinstance(results['silhouette_scores'], dict)
    
    # Test with all diagnostics
    all_results = clusters.display_diagnostics()
    assert 'silhouette_scores' in all_results
    assert 'inertia_scores' in all_results

def test_display_diagnostics_without_clustering(test_df, column_mapping):
    """Test diagnostics without running clustering first"""
    clusters = CustomerClusters(test_df, column_mapping)
    with pytest.raises(ValueError, match="Must run rfm_clusters"):
        clusters.display_diagnostics()

def test_save_and_load_model(test_df, column_mapping, tmp_path):
    """Test model saving and loading functionality"""
    # Create and train a model
    clusters = CustomerClusters(test_df, column_mapping)
    cols_to_transform = ['monetary', 'frequency']
    clusters.rfm_clusters(cols_to_transform, best_k=2)
    
    # Save the model
    model_path = tmp_path / "model.pkl"
    clusters.save_model(str(model_path))
    assert model_path.exists()
    
    # Load the model
    loaded_clusters = CustomerClusters.load_model(
        str(model_path),
        test_df,
        column_mapping
    )
    
    assert loaded_clusters.model is not None
    assert loaded_clusters.scaler is not None
    assert isinstance(loaded_clusters.config, ClusterConfig)

def test_save_model_without_training(test_df, column_mapping, tmp_path):
    """Test saving model without training first"""
    clusters = CustomerClusters(test_df, column_mapping)
    model_path = tmp_path / "model.pkl"
    
    with pytest.raises(ValueError, match="No trained model to save"):
        clusters.save_model(str(model_path))

def test_custom_config(test_df, column_mapping):
    """Test initialization with custom configuration"""
    custom_config = DiagnosticConfig(
        min_clusters=3,
        max_clusters=8,
        random_state=123,
        best_clusters=4
    )
    
    clusters = CustomerClusters(test_df, column_mapping, config=custom_config)
    assert clusters.config.min_clusters == 3
    assert clusters.config.max_clusters == 8
    assert clusters.config.random_state == 123
    assert clusters.config.best_clusters == 4