import sys
import os
import pytest
import pandas as pd
import numpy as np
from typing import Dict

sys.path.append(os.path.abspath(""))
# print("Current sys.path:", sys.path)

from shared.core.data_class import CustomDataFrame

TEST_FILE = 'pytest_data.csv'

# Fixture for test data
@pytest.fixture
def test_df():
    """Create a test DataFrame fixture."""
    return CustomDataFrame.read_csv(TEST_FILE)

# Test class inheritance and type
def test_inheritance(test_df):
    """Test that CustomDataFrame properly inherits from pandas DataFrame."""
    assert isinstance(test_df, CustomDataFrame)
    assert isinstance(test_df, pd.DataFrame)

# Test DataFrame operations maintain CustomDataFrame type
def test_dataframe_operations(test_df):
    """Test that DataFrame operations maintain CustomDataFrame type."""
    # Test basic operations
    subset = test_df.head()
    filtered = test_df[test_df.columns[0:2]]
    
    assert isinstance(subset, CustomDataFrame)
    assert isinstance(filtered, CustomDataFrame)

# Test validate method
class TestValidateMethod:
    """Test cases for the validate method."""
    
    def test_validate_basic(self, test_df):
        """Test basic validation functionality."""
        result = test_df.validate('pytest')
        assert isinstance(result, Dict)
        assert 'pytest' in result
        assert isinstance(result['pytest'], CustomDataFrame)
        
    def test_validate_errors(self, test_df):
        """Test validate method error handling."""
        # Test missing name parameter
        with pytest.raises(ValueError):
            test_df.validate()
            
        # Test invalid name type
        with pytest.raises(TypeError):
            test_df.validate(1)
            
        # Test invalid display_df option
        with pytest.raises(ValueError):
            test_df.validate('pytest', display_df='InvalidOption')
    
    def test_validate_display_options(self, test_df):
        """Test different display options."""
        valid_options = ['Info', 'Describe', 'Head', 'Sample']
        for option in valid_options:
            result = test_df.validate('pytest', display_df=option)
            assert isinstance(result, Dict)
            assert 'pytest' in result

# Test read_csv method
def test_read_csv():
    """Test the read_csv class method."""
    df = CustomDataFrame.read_csv(TEST_FILE)
    assert isinstance(df, CustomDataFrame)
    assert len(df) > 0

# Test analyze_statistics method
class TestAnalyzeStatistics:
    """Test cases for the analyze_statistics method."""
    
    def test_analyze_statistics_basic(self, test_df):
        """Test basic statistics analysis functionality."""
        num_stats, cat_stats = test_df.analyze_statistics('pytest')
        
        assert isinstance(num_stats, CustomDataFrame)
        assert isinstance(cat_stats, CustomDataFrame)
        
    def test_analyze_statistics_parameters(self, test_df):
        """Test analyze_statistics with different parameters."""
        # Test with different decimal places
        num_stats, cat_stats = test_df.analyze_statistics('pytest', decimal_places=3)
        
        # Test with different target column
        if len(test_df.columns) > 1:
            num_stats, cat_stats = test_df.analyze_statistics('pytest', 
                                                            target_col=test_df.columns[0])

# Test data integrity
def test_data_integrity(test_df):
    """Test that CustomDataFrame maintains data integrity."""
    # Test index
    assert test_df.index.equals(pd.DataFrame(test_df).index)
    
    # Test columns
    assert all(test_df.columns == pd.DataFrame(test_df).columns)
    
    # Test values
    assert np.array_equal(test_df.values, pd.DataFrame(test_df).values)

if __name__ == '__main__':
    pytest.main([__file__])