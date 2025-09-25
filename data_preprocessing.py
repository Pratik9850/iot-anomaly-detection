"""
IoT Anomaly Detection - Data Preprocessing Module

This module handles feature engineering and scaling for IoT sensor data.
It includes functions to:
- Load and clean IoT sensor data from various sources
- Apply feature engineering techniques like rolling windows and statistical features
- Normalize and scale data using StandardScaler or MinMaxScaler
- Handle missing values and outliers
- Create time-based features from timestamps
- Split data into training and validation sets for anomaly detection

Usage:
    from data_preprocessing import IoTDataPreprocessor
    preprocessor = IoTDataPreprocessor()
    X_scaled, scaler = preprocessor.fit_transform(raw_data)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime, timedelta

class IoTDataPreprocessor:
    """
    Preprocessor for IoT sensor data with feature engineering and scaling capabilities.
    """
    
    def __init__(self, scaler_type='standard', window_size=10):
        """
        Initialize the preprocessor with specified scaler and window size.
        
        Args:
            scaler_type (str): Type of scaler ('standard' or 'minmax')
            window_size (int): Window size for rolling statistics
        """
        self.scaler_type = scaler_type
        self.window_size = window_size
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.feature_columns = None
        
    def create_time_features(self, df, timestamp_col='timestamp'):
        """
        Create time-based features from timestamp column.
        
        Args:
            df (pd.DataFrame): Input dataframe with timestamp column
            timestamp_col (str): Name of timestamp column
            
        Returns:
            pd.DataFrame: DataFrame with additional time features
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time components
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_year'] = df[timestamp_col].dt.dayofyear
        df['month'] = df[timestamp_col].dt.month
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def create_rolling_features(self, df, sensor_columns):
        """
        Create rolling window statistical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            sensor_columns (list): List of sensor column names
            
        Returns:
            pd.DataFrame: DataFrame with rolling features
        """
        df = df.copy()
        
        for col in sensor_columns:
            # Rolling statistics
            df[f'{col}_rolling_mean'] = df[col].rolling(window=self.window_size).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=self.window_size).std()
            df[f'{col}_rolling_min'] = df[col].rolling(window=self.window_size).min()
            df[f'{col}_rolling_max'] = df[col].rolling(window=self.window_size).max()
            
            # Lag features
            df[f'{col}_lag_1'] = df[col].shift(1)
            df[f'{col}_lag_3'] = df[col].shift(3)
            
            # Rate of change
            df[f'{col}_diff'] = df[col].diff()
            df[f'{col}_pct_change'] = df[col].pct_change()
        
        return df
    
    def handle_missing_values(self, df, strategy='interpolate'):
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Strategy to handle missing values
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        df = df.copy()
        
        if strategy == 'interpolate':
            df = df.interpolate(method='time')
        elif strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif strategy == 'drop':
            df = df.dropna()
        
        return df
    
    def detect_and_handle_outliers(self, df, sensor_columns, method='iqr', factor=1.5):
        """
        Detect and handle outliers using IQR method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            sensor_columns (list): List of sensor columns
            method (str): Outlier detection method
            factor (float): IQR factor for outlier detection
            
        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        df = df.copy()
        
        for col in sensor_columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                # Cap outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def fit_transform(self, df, sensor_columns, timestamp_col='timestamp'):
        """
        Fit the preprocessor and transform the data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            sensor_columns (list): List of sensor column names
            timestamp_col (str): Name of timestamp column
            
        Returns:
            tuple: Scaled features and fitted scaler
        """
        # Create time features
        df_processed = self.create_time_features(df, timestamp_col)
        
        # Create rolling features
        df_processed = self.create_rolling_features(df_processed, sensor_columns)
        
        # Handle missing values
        df_processed = self.handle_missing_values(df_processed)
        
        # Handle outliers
        df_processed = self.detect_and_handle_outliers(df_processed, sensor_columns)
        
        # Select feature columns (exclude timestamp and non-numeric columns)
        feature_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if timestamp_col in feature_columns:
            feature_columns.remove(timestamp_col)
        
        self.feature_columns = feature_columns
        X = df_processed[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, self.scaler
    
    def transform(self, df, sensor_columns, timestamp_col='timestamp'):
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df (pd.DataFrame): Input dataframe
            sensor_columns (list): List of sensor column names
            timestamp_col (str): Name of timestamp column
            
        Returns:
            np.ndarray: Scaled features
        """
        if self.scaler is None or self.feature_columns is None:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # Apply same preprocessing steps
        df_processed = self.create_time_features(df, timestamp_col)
        df_processed = self.create_rolling_features(df_processed, sensor_columns)
        df_processed = self.handle_missing_values(df_processed)
        df_processed = self.detect_and_handle_outliers(df_processed, sensor_columns)
        
        # Use the same feature columns as during fitting
        X = df_processed[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled

def load_iot_data(file_path, timestamp_col='timestamp'):
    """
    Load IoT data from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        timestamp_col (str): Name of timestamp column
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(file_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col)
    return df

def create_sequences(X, y=None, sequence_length=10):
    """
    Create sequences for time series anomaly detection.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values (optional)
        sequence_length (int): Length of sequences
        
    Returns:
        tuple: Sequences and targets (if provided)
    """
    sequences = []
    targets = [] if y is not None else None
    
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:(i + sequence_length)])
        if y is not None:
            targets.append(y[i + sequence_length - 1])
    
    sequences = np.array(sequences)
    if targets is not None:
        targets = np.array(targets)
        return sequences, targets
    
    return sequences

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Example data creation for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 20 + np.random.normal(0, 2, 1000),
        'humidity': 50 + np.random.normal(0, 5, 1000),
        'pressure': 1013 + np.random.normal(0, 10, 1000),
        'vibration': np.random.exponential(1, 1000)
    })
    
    # Initialize preprocessor
    preprocessor = IoTDataPreprocessor(scaler_type='standard', window_size=5)
    
    # Define sensor columns
    sensor_cols = ['temperature', 'humidity', 'pressure', 'vibration']
    
    # Fit and transform data
    X_scaled, scaler = preprocessor.fit_transform(sample_data, sensor_cols)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Scaled data shape: {X_scaled.shape}")
    print(f"Feature columns: {preprocessor.feature_columns}")
    
    # Create sequences for time series modeling
    sequences = create_sequences(X_scaled, sequence_length=10)
    print(f"Sequences shape: {sequences.shape}")
