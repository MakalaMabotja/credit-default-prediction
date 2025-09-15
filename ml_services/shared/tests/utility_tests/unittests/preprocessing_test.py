import os 
import sys 

sys.path.append(os.path.abspath(""))

import pytest
import pandas as pd
import numpy as np 

from shared.utils import (id_transform, 
                            cusomer_map,
                            lender_loan_map, 
                            date_treatment,
                            analyze_refinancing,
                            bin_features
                            )

from shared.core import CustomDataFrame

# Replace with your test file
TEST_FILE = 'C:/Python Projects/2-CreditScore Predictions/african-credit-scoring-challenge20241129-14702-1nqro8v/Train.csv'

# Fixture for test data
@pytest.fixture
def test_df():
    """Create a test DataFrame fixture."""
    return CustomDataFrame.read_csv(TEST_FILE)

def test_id_transform(test_df):
    """Test ID column transformation from numeric to object type."""
    
    # Check test data has ID columns of numeric type
    id_columns = [col for col in test_df.columns if 'id' in col.lower()]
    assert id_columns, "Test data must contain ID columns"
    assert any(test_df[col].dtype == np.int64 for col in id_columns), "ID columns should start as numeric"

    # Transform and verify
    transformed = id_transform(test_df)
    assert all(transformed[col].dtype == np.dtype('O') for col in id_columns), "ID columns should be converted to object type"

def test_customer_map(test_df):
    """Test mapping of new versus repeat customers"""
    
    # Check that data contain label for new and repet loans
    assert (test_df['New_versus_Repeat'].dtype == np.dtype('O'))
    assert set(test_df['New_versus_Repeat'].unique()) == {'New Loan', 'Repeat Loan'}
    
    # check transformation completed properly
    transformed = cusomer_map(test_df)
    assert isinstance(transformed, CustomDataFrame)
    assert (transformed['new'].dtype == np.dtype(int))

def test_lender_loan_map(test_df):
    """ Test for lender_id and lona_type mapping. Expecting numerical dtype to be converted to str"""
    assert (test_df['lender_id'].dtype == np.dtype(np.int64))
    
    # check transformation of lender mapping function
    transformed = lender_loan_map(test_df)
    assert isinstance(transformed, CustomDataFrame)
    assert (transformed['lender_id'].dtype == np.dtype('O'))
    assert (transformed['loan_type'].dtype == np.dtype('O'))
    
    # check that ids have been mapped into alphabetical values
    assert (set(['lender_A', 'lender_B']).issubset(transformed['lender_id'].unique()))
    assert (set(['A', 'B']).issubset(transformed['loan_type'].unique()))

def test_lender_loan_calculations(test_df):
    """ Tes to ensure the calculation of repay conditions are accurate """
    transformed = lender_loan_map(test_df)
    assert (set(['pct_repay', 'repay_rate']).issubset(transformed.columns))
    assert all(transformed['pct_repay'] == transformed['Total_Amount_to_Repay'] / transformed['Total_Amount'])
    assert all(transformed['repay_rate'] == transformed['Total_Amount_to_Repay'] / transformed['duration'])

@pytest.fixture
def date_df ():
    """ pyTest fixture to test date transformation functionality with predictable results"""
    return CustomDataFrame({
        'customer_id': [1,2,2,3,1,4,5,3],
        'disbursement_date': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-02-01', '2023-01-01', '2023-01-05', '2023-01-10', '2023-02-01']
        })
    
def test_date_treatment_column_values(date_df):
    """ Test date time format of disbursement column and cohort value"""
    assert (date_df['disbursement_date'].dtype == np.dtype('O'))
    date_transformed = date_treatment(date_df)
    assert isinstance(date_transformed, CustomDataFrame)
    assert pd.api.types.is_datetime64_ns_dtype(date_transformed['disbursement_date'])
    
    expected_cohort = pd.to_datetime(date_df['disbursement_date']).dt.strftime('%Y-%m').tolist()
    for idx, cohort in enumerate(date_transformed['cohort']):
        assert cohort == expected_cohort[idx]
        
def test_date_treatment_value_validation(date_df):
    """ Test accurate """
    date_transformed = date_treatment(date_df)
    
    assert (date_transformed.loc[ 1, 'dis_year'] == 2023)
    assert (date_transformed.loc[ 1, 'dis_month'] == 1)
    assert (date_transformed.loc[ 1, 'dis_day'] == 5)
    assert (date_transformed.loc[ 1, 'dis_day_of_week'] == 3)
    assert (date_transformed.loc[ 1, 'dis_day_of_year'] == 5)
    
    
def test_days_since_last_loan(date_df):
    date_transformed = date_treatment(date_df)
    assert pd.api.types.is_integer_dtype(date_transformed['days_since_last_loan'])
    assert (date_transformed.loc[1, 'days_since_last_loan'] == 0)


def test_date_treatment_functionality(test_df):
    original_idx = test_df.index.tolist()
    transdformed = date_treatment(test_df)
    
    assert isinstance(transdformed, CustomDataFrame)
    assert (transdformed.index.tolist() == original_idx)
    

def test_date_columns_added(test_df):
    
    transformed = date_treatment(test_df)
    expected_columns = ['dis_year', 'dis_month', 'dis_day', 'dis_day_of_week', 
                        'dis_day_of_year', 'cohort', 'days_since_last_loan']
    for col in expected_columns:
        assert col in transformed.columns

def test_analyze_refinancing_validation(test_df):
    """ Test to ensure that correct data trnaformations are done"""
    
    # ensure correct dataframe type is returned
    transformed = analyze_refinancing(test_df)
    assert isinstance(transformed, CustomDataFrame)
    
    # ensure correct order is reuturned after transformation
    assert all(transformed.index == test_df.index)
    
    # ensure no nulls are returned after transformation
    assert (transformed.isna().sum().sum() == 0)
    
    # ensure merging does not create duplicate entries
    assert (transformed.duplicated().sum().sum() == 0)
    
    
def test_analyze_refinancing_functionality(test_df):
    """ Test correct columns added """
    
    # test dataframe must contain duplicate loan ids
    assert (test_df['tbl_loan_id'].duplicated().sum() > 0)
    
    # test that correct columsn created
    transformed = analyze_refinancing(test_df)
    required_cols = ['refinanced', 'refinance_amount', 'increased_risk']
    for col in required_cols:
        assert (col in transformed.columns)
        
@pytest.fixture
def finance_df():
    """ finance dataframe to test calculation by refinance function"""
    return CustomDataFrame({
        'customer_id' : [1,2,1,2,4,5],
        'lender_id': [1,1,1,2,3,2],
        'tbl_loan_id': [1,2,1,2,4,5],
        'Amount_Funded_By_Lender' : [100, 150, 200, 100, 50, 50],
        'Lender_portion_Funded' : [0.1, 0.2, 0.1, 0.25, 0, 0]        
    })

def test_refinance_values(finance_df):
    """Test refinancing calculations."""
    # Apply the function
    result_df = analyze_refinancing(finance_df)

    # Assertions for customer_id == 1
    customer_1 = result_df[result_df['customer_id'] == 1].reset_index()
    assert customer_1.loc[0, 'refinanced'] == 0
    assert customer_1.loc[1, 'refinanced'] == 0
    assert customer_1['refinance_amount'].sum() == 0

    # Assertions for customer_id == 2
    customer_2 = result_df[result_df['customer_id'] == 2].reset_index()
    assert customer_2.loc[0 ,'refinanced'] == 1
    assert customer_2.loc[0, 'refinance_amount'] == -50
    assert all(np.isclose(customer_2['increased_risk'], 0.05))
    
@pytest.fixture
def bin_df():
    """Fixture to provide a sample DataFrame for testing."""
    return CustomDataFrame({
        'lender_portion': [0.1, 0.25, 0.6, 1.5, np.nan],
        'duration': [5, 30, 365, 720, np.nan],
        'Total_Amount_to_Repay': [50, 500, 15000, 100000, np.nan],
        'target': [1, 0, 1, 1, 0],
    })

def test_basic_binning(bin_df):
    """Test basic binning functionality."""
    result_df = bin_features(bin_df, vectorize_bins=False)
    
    # Check new binned columns exist
    expected_columns = ['binned_lender_portion', 'binned_duration', 'binned_amounts']
    for col in expected_columns:
        assert col in result_df.columns, f"{col} is missing in the result DataFrame."
    
    # Validate binned column types
    for col in expected_columns:
        assert isinstance(result_df[col].dtype, pd.CategoricalDtype), f"{col} is not of categorical dtype."

def test_vectorization_enabled(bin_df):
    """Test vectorization functionality with bin_score."""
    result_df = bin_features(bin_df, vectorize_bins=True)
    
    # Check bin_score column exists
    assert 'bin_score' in result_df.columns, "bin_score column is missing when vectorization is enabled."
    
    # Validate bin_score values are computed correctly
    bin_score_mapping = result_df.groupby(['binned_lender_portion', 'binned_duration', 'binned_amounts'], observed=False)['target'].mean().to_dict()
    for _, row in result_df.iterrows():
        if not pd.isnull(row['binned_lender_portion']) and not pd.isnull(row['binned_duration']) and not pd.isnull(row['binned_amounts']):
            bin_combination = (row['binned_lender_portion'], row['binned_duration'], row['binned_amounts'])
            assert row['bin_score'] == pytest.approx(bin_score_mapping[bin_combination]), \
                f"bin_score for {bin_combination} is incorrect."

def test_vectorization_disabled(bin_df):
    """Test disabling vectorization."""
    result_df = bin_features(bin_df, vectorize_bins=False)
    
    # Ensure bin_score column does not exist
    assert 'bin_score' not in result_df.columns, "bin_score column should not exist when vectorization is disabled."
    
def test_missing_values(bin_df):
    """Test behavior with NaN values."""
    result_df = bin_features(bin_df, vectorize_bins=True)
    
    # Check that NaN values in original columns do not break binning
    assert result_df['binned_lender_portion'].isnull().sum() == 1, "NaN handling for binned_lender_portion is incorrect."
    assert result_df['binned_duration'].isnull().sum() == 1, "NaN handling for binned_duration is incorrect."
    assert result_df['binned_amounts'].isnull().sum() == 1, "NaN handling for binned_amounts is incorrect."

def test_empty_dataframe():
    """Test behavior with an empty DataFrame."""
    empty_df = CustomDataFrame(columns=['lender_portion', 'duration', 'Total_Amount_to_Repay', 'target'])
    result_df = bin_features(empty_df)
    
    # Ensure result is still a CustomDataFrame and remains empty
    assert isinstance(result_df, CustomDataFrame), "Result is not a CustomDataFrame."
    assert result_df.empty, "Result DataFrame should be empty."

def test_extreme_values():
    """Test behavior with extreme values."""
    extreme_df = CustomDataFrame({
        'lender_portion': [-10, 2],  # Values outside bin ranges
        'duration': [-1, 10000],    # Values outside bin ranges
        'Total_Amount_to_Repay': [-100, 1000000],  # Values outside bin ranges
        'target': [0, 1]
    })
    result_df = bin_features(extreme_df, vectorize_bins=False)
    
    # Check that extreme values are binned correctly
    assert result_df['binned_lender_portion'].isnull().sum() == 1, "Extreme lender_portion values not handled correctly."
    assert result_df['binned_duration'].isnull().sum() == 1, "Extreme duration values not handled correctly."
    assert result_df['binned_amounts'].isnull().sum() == 1, "Extreme Total_Amount_to_Repay values not handled correctly."

