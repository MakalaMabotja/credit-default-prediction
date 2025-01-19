import pandas as pd
import numpy as np 

import sys
import os

sys.path.append(os.path.abspath(""))
# print("Current sys.path:", sys.path)

from shared.core.data_class import CustomDataFrame

def id_transform(df: CustomDataFrame) -> CustomDataFrame:
    """
    Converts all columns containing 'id' in their names to the 'object' dtype.

    Parameters:
    df (CustomDataFrame): Input DataFrame.

    Returns:
    CustomDataFrame: DataFrame with 'id' columns converted to 'object'.
    """
    # Create a dictionary of columns with 'id' and their target dtype
    id_maps = {col: 'object' for col in df.columns if 'id' in col}
    
    for col, dtype in id_maps.items():
        df[col] = df[col].astype(dtype)
    
    return df

def cusomer_map(df: CustomDataFrame) -> CustomDataFrame:
    # Map customer type
    customer_map = {value: idx for idx, value in enumerate(df['New_versus_Repeat'].unique())}
    df['new'] = df['New_versus_Repeat'].map(customer_map).astype(int)
    return df
            

def lender_loan_map(df: CustomDataFrame) -> CustomDataFrame:
    """
    Maps lender IDs and loan types in the DataFrame to simplified or anonymized representations.
    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing at least the columns 'lender_id' and 'loan_type'.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the 'lender_id' column mapped to a format like 'lender_A', 'lender_B', etc.,
        and the 'loan_type' column mapped to a single-character representation based on its numeric suffix.
    """
    
    #  mapping for 'lender_id' 
    lender_map = {
        name: f'lender_{chr(idx + ord("A"))}'  # chr(idx + ord('A')) converts 0 -> 'A', 1 -> 'B', etc.
        for idx, name in enumerate(df['lender_id'].unique())
    }

    #  mapping for 'loan_type' 
    loan_map = {
        loan_type: chr(ord('A') + int(loan_type.split('_')[1]) - 1)  # Extracts the numeric part and converts it to 'A', 'B', etc.
        for loan_type in df['loan_type'].unique()
    }

    df['lender_id'] = df['lender_id'].map(lender_map)
    df['loan_type'] = df['loan_type'].map(loan_map)
    
    df['pct_repay'] = df['Total_Amount_to_Repay'] / df['Total_Amount']
    df['repay_rate'] = df['Total_Amount_to_Repay'] / df['duration']
    
    return df

def date_treatment(df: CustomDataFrame) -> CustomDataFrame:
    """
    Processes the 'disbursement_date' column in the DataFrame to extract various date components
    and create additional features related to the date.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing at least the 'disbursement_date' column with date values.

    Returns:
    --------
    pandas.DataFrame
        The original DataFrame with added columns for year, month, day, day of the week, 
        day of the year, and a cohort identifier in 'YYYY-MM' format, and the days since the last loan..
    """
    
    df['disbursement_date'] = pd.to_datetime(df['disbursement_date'])
    df['dis_year'] = df['disbursement_date'].dt.year
    df['dis_month'] = df['disbursement_date'].dt.month
    df['dis_day'] = df['disbursement_date'].dt.day
    df['dis_day_of_week'] = df['disbursement_date'].dt.day_of_week
    df['dis_day_of_year'] = df['disbursement_date'].dt.day_of_year
    df['cohort'] = df['disbursement_date'].dt.strftime('%Y-%m')
    
    df = df.sort_values(by=['customer_id', 'disbursement_date'])
    df['days_since_last_loan'] = (
    df.groupby('customer_id')['disbursement_date']
        .diff()
        .dt.days
        .fillna(0)
        .astype(int)
    )
    
    return df.sort_index()

def analyze_refinancing(df: CustomDataFrame) -> CustomDataFrame:
    """
    Analyzes loan refinancing patterns by identifying duplicate loan IDs and calculating
    refinancing metrics when the lender changes.
    
    Parameters:
    df (pandas.DataFrame): DataFrame containing loan data with columns:
        - tbl_loan_id: Unique loan identifier
        - lender_id: Identifier for the lender
        - Amount_Funded_By_Lender: Loan amount
        - Lender_portion_Funded: Risk portion
        - customer_id: Unique customer identifier
    
    Returns:
    pandas.DataFrame: Original dataframe with added columns:
        - refinanced: Binary indicator (1 if refinanced)
        - refinance_amount: Change in loan amount after refinancing
        - increased_risk: Change in lender risk portion
    """    
    df_copy = df.reset_index()
    duplicate_loan_ids = df_copy['tbl_loan_id'].value_counts()[df_copy['tbl_loan_id'].value_counts() > 1].index
    # duplicate_loan_ids = df_copy['tbl_loan_id'][df_copy['tbl_loan_id'].duplicated(keep=False)]

    filtered_df = (df_copy[df_copy['tbl_loan_id'].isin(duplicate_loan_ids)].sort_values(['tbl_loan_id', 'customer_id']))

    refinance_col = ['refinanced', 'refinance_amount','increased_risk']
    for col in refinance_col:
            filtered_df[col] = np.nan

    shifted_df = filtered_df.shift(-1)

    same_loan_mask = filtered_df['tbl_loan_id'] == shifted_df['tbl_loan_id']
    different_lender_mask = filtered_df['lender_id'] != shifted_df['lender_id']
    refinance_mask = same_loan_mask & different_lender_mask

    # Calculate refinancing metrics
    filtered_df.loc[refinance_mask, 'refinanced'] = 1
    filtered_df.loc[refinance_mask, 'refinance_amount'] = (
        shifted_df.loc[refinance_mask,'Amount_Funded_By_Lender'] - 
        filtered_df.loc[refinance_mask,'Amount_Funded_By_Lender']
    )
    filtered_df.loc[refinance_mask, 'increased_risk'] = (
        shifted_df.loc[refinance_mask,'Lender_portion_Funded'] - 
        filtered_df.loc[refinance_mask,'Lender_portion_Funded']
    )
    filtered_df['refinanced'] = filtered_df['refinanced'].ffill( axis=0)
    filtered_df['refinance_amount'] = filtered_df['refinance_amount'].ffill( axis=0)
    filtered_df['increased_risk'] = filtered_df['increased_risk'].ffill( axis=0)

    merged_df = pd.concat([df_copy, filtered_df[refinance_col]], axis=1)

    merged_df.set_index('index', inplace=True)
    # merged_df.index.name=None
    merged_df.fillna(0, inplace=True)

    return merged_df

def bin_features(df: CustomDataFrame, vectorize_bins: bool = True) -> CustomDataFrame:
    """
    Bin numerical features into categorical ranges and optionally vectorize bins based on target variable statistics.

    Parameters
    ----------
    df : CustomDataFrame
        The input DataFrame containing the following required columns:
        - 'lender_portion': Numerical column to be binned based on `portion_bins`.
        - 'duration': Numerical column to be binned based on `duration_bins`.
        - 'Total_Amount_to_Repay': Numerical column to be binned based on `amount_bins`.
        - 'target': Numerical column representing the target variable for calculating mean statistics (used if `vectorize_bins=True`).

    vectorize_bins : bool, optional (default=True)
        If True, computes the mean target value for each combination of binned categories and maps them to a new column `bin_score`.

    Returns
    -------
    CustomDataFrame
        The input DataFrame is return is 
    Notes
    -----
    - The feature bins are predefined ranges used to discretize the numerical features.
    """

    #Feature bins
    duration_bins = [0,6,7,14,30,90,180,360,540,720, np.inf]
    portion_bins = [0, 0.12, 0.2,0.25, 0.3, 0.37, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, np.inf]
    amount_bins = [0, 10, 100, 315, 1000, 2295, 5000, 7500, 10000, 11450, 15000,25000, 50000, 100000, 119411, np.inf]

    df['binned_lender_portion'] = pd.cut(df['lender_portion'], bins=portion_bins)
    df['binned_duration'] = pd.cut(df['duration'], bins=duration_bins)
    df['binned_amounts'] = pd.cut(df['Total_Amount_to_Repay'], bins=amount_bins)
    
    if vectorize_bins:
        # Calculate mean target for each combination
        bin_summary = df.groupby(['binned_lender_portion', 'binned_duration', 'binned_amounts'], observed=False)['target'].mean()
        
        # Create a mapping dictionary from bin combinations to their mean target
        bin_mapping = bin_summary.to_dict()
        
        df['bin_combination'] = df.apply(
            lambda x: (x['binned_lender_portion'], 
                      x['binned_duration'], 
                      x['binned_amounts']), 
            axis=1
        )
        
        # Map the combinations to their mean target values
        df['bin_score'] = df['bin_combination'].map(bin_mapping)
        df = df.drop('bin_combination', axis=1)
    
    return df