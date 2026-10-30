import numpy as np
import pandas as pd
from src.config import CATEGORICAL_COLS

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw network flow dataframe:
    - Strips whitespace from strings.
    - Standardizes missing representations ('-', '', 'null').
    - Replaces infinite values with NaN and imputes medians.
    """
    df = df.copy()
    
    # Clean string columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'-': 'missing', '': 'missing', 'nan': 'missing'})
            
    # Handle infinities in numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill any NaNs with 0 (network traffic counts/rates default to 0 if absent)
    df[num_cols] = df[num_cols].fillna(0)
    
    return df
