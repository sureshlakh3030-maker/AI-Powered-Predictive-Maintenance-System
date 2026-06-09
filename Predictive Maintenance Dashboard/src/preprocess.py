import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading dataset: {str(e)}")

def sanitize_columns(df):
    """
    Rename columns to remove special characters forbidden by XGBoost (like [, ] or <)
    and replace spaces with underscores for cleaner names.
    
    Parameters:
    df (pd.DataFrame): Input dataframe.
    
    Returns:
    pd.DataFrame: Dataframe with sanitized column names.
    """
    df = df.copy()
    
    # Direct mapping for original dataset columns
    rename_dict = {
        'Air temperature [K]': 'Air_temperature_K',
        'Process temperature [K]': 'Process_temperature_K',
        'Rotational speed [rpm]': 'Rotational_speed_rpm',
        'Torque [Nm]': 'Torque_Nm',
        'Tool wear [min]': 'Tool_wear_min',
        'Machine failure': 'Machine_failure'
    }
    
    df = df.rename(columns=rename_dict)
    
    # General cleanup for any remaining columns to remove [, ], <, >
    new_cols = []
    for col in df.columns:
        c = col.replace('[', '').replace(']', '').replace('<', '').replace('>', '').strip().replace(' ', '_')
        new_cols.append(c)
    df.columns = new_cols
    
    return df

def clean_data(df):
    """
    Handle missing values, duplicate rows, and remove unnecessary identifier columns.
    
    Parameters:
    df (pd.DataFrame): Raw dataframe.
    
    Returns:
    pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()
    
    # 1. Null value checking
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        df = df.dropna()
        
    # 2. Duplicate removal
    df = df.drop_duplicates()
    
    # 3. Drop UDI and Product ID
    columns_to_drop = ['UDI', 'Product ID']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df

def encode_type(df):
    """
    Encode the 'Type' column. Map 'L' (Low), 'M' (Medium), 'H' (High) to integers 0, 1, 2.
    
    Parameters:
    df (pd.DataFrame): Dataframe with 'Type' column.
    
    Returns:
    pd.DataFrame: Dataframe with encoded 'Type' column.
    """
    df = df.copy()
    if 'Type' in df.columns:
        type_mapping = {'L': 0, 'M': 1, 'H': 2}
        df['Type'] = df['Type'].map(type_mapping)
        # Fill any unknown values with a default category, just in case
        df['Type'] = df['Type'].fillna(-1).astype(int)
    return df

def engineer_features(df):
    """
    Create custom features relevant to predictive maintenance:
    - temp_diff = Process Temperature - Air Temperature
    - power = RPM * Torque
    - stress = Torque * Tool Wear
    
    Parameters:
    df (pd.DataFrame): Dataframe with base features (already sanitized).
    
    Returns:
    pd.DataFrame: Dataframe with engineered features.
    """
    df = df.copy()
    
    # Required base columns (sanitized names)
    air_temp_col = 'Air_temperature_K'
    proc_temp_col = 'Process_temperature_K'
    rpm_col = 'Rotational_speed_rpm'
    torque_col = 'Torque_Nm'
    tool_wear_col = 'Tool_wear_min'
    
    # Check if necessary columns exist
    required_cols = [air_temp_col, proc_temp_col, rpm_col, torque_col, tool_wear_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")
        
    # temp_diff = Process Temperature - Air Temperature
    df['temp_diff'] = df[proc_temp_col] - df[air_temp_col]
    
    # power = RPM * Torque
    df['power'] = df[rpm_col] * df[torque_col]
    
    # stress = Torque * Tool Wear
    df['stress'] = df[torque_col] * df[tool_wear_col]
    
    return df

def preprocess_pipeline(file_path):
    """
    Executes the entire data preprocessing and split pipeline.
    
    Parameters:
    file_path (str): Path to raw CSV file.
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, and feature list.
    """
    # Load
    df = load_data(file_path)
    
    # Clean
    df = clean_data(df)
    
    # Sanitize Columns (removes [K], [Nm], [rpm], [min], spaces)
    df = sanitize_columns(df)
    
    # Encode
    df = encode_type(df)
    
    # Feature Engineering
    df = engineer_features(df)
    
    # Define Target and Feature list
    target_col = 'Machine_failure'
    
    # Drop specific failure types (TWF, HDF, PWF, OSF, RNF) since they are leakage columns
    # that directly reveal if a failure happened and why.
    leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    y = df[target_col]
    
    feature_cols = [col for col in df.columns if col not in [target_col] + leakage_cols]
    X = df[feature_cols]
    
    # Stratified split to preserve failure ratio (dataset is highly imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    print("Testing Preprocessing Pipeline...")
    try:
        # Default local path
        data_path = os.path.join("Data", "ai4i2020.csv")
        X_train, X_test, y_train, y_test, features = preprocess_pipeline(data_path)
        print("Success!")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Target distribution in Train:\n{y_train.value_counts(normalize=True)}")
        print(f"Features: {features}")
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
