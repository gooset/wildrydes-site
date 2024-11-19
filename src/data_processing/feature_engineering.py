import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame, target_col: str = 'arbovirus_bool'):
        """
        Initialize the feature engineer with input data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe containing required columns
        target_col : str
            Name of the target variable column
        """
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.target_col = target_col
        self.selected_features = None
        
    def create_temporal_features(self):
        """
        Generate temporal features with cyclic encoding
        """
        # Basic temporal extraction
        self.df['month'] = self.df['date'].dt.month
        self.df['year'] = self.df['date'].dt.year
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
        # Seasonal classification
        self.df['season'] = pd.cut(
            self.df['month'],
            bins=[0, 5, 10, 12], 
            labels=['Dry', 'Rainy', 'Dry'], 
            right=True, 
            ordered=False
        )
        
        # Cyclic encoding for temporal periodicity
        for col, max_val in [
            ('month', 12), 
            ('day_of_year', 365),
            ('week_of_year', 52), 
            ('day_of_week', 7)
        ]:
            self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / max_val)
            self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / max_val)
        
        return self
    
    def create_weather_features(self):
        """
        Generate advanced weather-related features with dynamic window selection
        """
        # Base weather columns
        weather_cols = [
            'temperature_2m_max', 'temperature_2m_min', 
            'precipitation_sum', 'wind_speed_10m_max'
        ]
        
        # Create initial rolling statistics with various windows
        for col in weather_cols:
            # Create rolling statistics with multiple windows
            for window in range(3, 15, 2):  # Odd windows from 3 to 14
                self.df[f'{col}_{window}d_mean'] = (
                    self.df.groupby('city')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                self.df[f'{col}_{window}d_std'] = (
                    self.df.groupby('city')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )
        
        # Disease-specific weather indicators
        self.df['optimal_mosquito_temp'] = (
            (self.df['temperature_2m_max'] >= 25) & 
            (self.df['temperature_2m_max'] <= 35)
        ).astype(int)
        
        self.df['breeding_conditions'] = (
            (self.df['precipitation_sum'] > 0) & 
            (self.df['temperature_2m_min'] > 20)
        ).astype(int)
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(method='ffill')
        self.df[numeric_cols] = self.df[numeric_cols].fillna(method='bfill')
        
        return self
    
    def remove_highly_correlated(self, threshold: float = 0.85) -> None:
        """
        Remove highly correlated features
        
        Parameters:
        -----------
        threshold : float
            Correlation threshold for feature elimination
        """
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        # Exclude non-feature columns
        exclude_cols = ['date', 'city', self.target_col]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = self.df[feature_cols].corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [
            column for column in upper_tri.columns 
            if any(upper_tri[column] > threshold)
        ]
        
        # Drop highly correlated features
        self.df = self.df.drop(columns=to_drop)
        
    def select_features(self, n_features: int = 10) -> List[str]:
        """
        Select features using Recursive Feature Elimination
        
        Parameters:
        -----------
        n_features : int
            Number of features to select
            
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        # Prepare feature matrix
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        exclude_cols = ['date', 'city', self.target_col]
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        X = self.df[feature_cols]
        y = self.df[self.target_col]
        
        # Initialize models
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=0.1
        )
        
        # Fit selector
        selector.fit(X, y)
        
        # Get selected features
        selected_features = [
            feature for feature, selected in zip(feature_cols, selector.support_)
            if selected
        ]
        
        self.selected_features = selected_features
        return selected_features
    
    def transform(self) -> pd.DataFrame:
        """
        Apply feature engineering pipeline and return transformed dataset
        
        Returns:
        --------
        pd.DataFrame
            Transformed dataset with selected features
        """
        # Create all features
        self.create_temporal_features()
        self.create_weather_features()
        
        # Remove highly correlated features
        self.remove_highly_correlated()
        
        # Select best features if not already done
        if self.selected_features is None:
            self.select_features()
        
        # Return dataset with selected features
        return_cols = ['city', 'date', self.target_col] + self.selected_features
        return self.df[return_cols]


# import pandas as pd
# import numpy as np

# class FeatureEngineer:
#     def __init__(self, df: pd.DataFrame):
#         """
#         Initialize feature engineering with input dataframe
        
#         Parameters:
#         -----------
#         df : pd.DataFrame
#             Input dataframe for feature engineering
#         """
#         self.df = df.copy()
#         self.df['date'] = pd.to_datetime(self.df['date'])
    
#     def create_temporal_features(self):
#         """
#         Generate comprehensive temporal features
        
#         Returns:
#         --------
#         FeatureEngineer
#             Instance with added temporal features
#         """
#         # Basic temporal extraction
#         self.df['month'] = self.df['date'].dt.month
#         self.df['year'] = self.df['date'].dt.year
#         self.df['day_of_year'] = self.df['date'].dt.dayofyear
#         self.df['week_of_year'] = self.df['date'].dt.isocalendar().week
#         self.df['day_of_week'] = self.df['date'].dt.dayofweek
        
#         # Seasonal classification
#         self.df['season'] = pd.cut(
#             self.df['month'],
#             bins=[0, 5, 10, 12], 
#             labels=['Dry', 'Rainy', 'Dry'], 
#             right=True, 
#             ordered=False
#         )
        
#         # Cyclic encoding for temporal periodicity
#         for col, max_val in [
#             ('month', 12), 
#             ('day_of_year', 365),
#             ('week_of_year', 52), 
#             ('day_of_week', 7)
#         ]:
#             self.df[f'{col}_sin'] = np.sin(2 * np.pi * self.df[col] / max_val)
#             self.df[f'{col}_cos'] = np.cos(2 * np.pi * self.df[col] / max_val)
        
#         return self
    
#     def create_weather_features(self):
#         """
#         Generate advanced weather-related features
        
#         Returns:
#         --------
#         FeatureEngineer
#             Instance with added weather features
#         """
#         # Default weather columns
#         weather_cols = [
#             'temperature_2m_max', 'temperature_2m_min', 
#             'precipitation_sum', 'wind_speed_10m_max'
#         ]
        
#         # Rolling window statistics
#         windows = [3, 7, 14]
#         for col in weather_cols:
#             for window in windows:
#                 self.df[f'{col}_{window}d_mean'] = (
#                     self.df.groupby('city')[col]
#                     .transform(lambda x: x.rolling(window, min_periods=1).mean())
#                 )
#                 self.df[f'{col}_{window}d_std'] = (
#                     self.df.groupby('city')[col]
#                     .transform(lambda x: x.rolling(window, min_periods=1).std())
#                 )
        
#     # Disease-specific weather indicators
#         self.df['optimal_mosquito_temp'] = (
#             (self.df['temperature_2m_max'] >= 25) & 
#             (self.df['temperature_2m_max'] <= 35)
#         ).astype(int)
        
#         self.df['breeding_conditions'] = (
#             (self.df['precipitation_sum'] > 0) & 
#             (self.df['temperature_2m_min'] > 20)
#         ).astype(int)
        
#         # Lag features for breeding conditions
#         lag_days = [3, 7, 14]
#         for lag in lag_days:
#             self.df[f'breeding_conditions_lag_{lag}'] = (
#                 self.df.groupby('city')['breeding_conditions']
#                 .shift(lag).fillna(0)
#             )

#         # List of columns non-weather-related
#         exclude_columns = [
#             'latitude', 'longitude', 'country', 'country_id', 
#             'status', 'occurrence_id', 'vector', 'source_type', 
#             'location_type', 'city', 'date'
#         ]

#         all_weather_columns = [col for col in self.df.columns if col not in exclude_columns]

#         # Apply forward fill and backward fill to the weather-related columns
#         self.df[all_weather_columns] = self.df[all_weather_columns].fillna(method='ffill')
#         self.df[all_weather_columns] = self.df[all_weather_columns].fillna(method='bfill')

#         # Drop rows where there are still missing values in essential columns
#         self.df.dropna(subset=exclude_columns, inplace=True)
#         return self


