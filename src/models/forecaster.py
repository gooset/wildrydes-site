import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

class DiseaseForecaster:
    def __init__(self, df):
        """Initialize forecaster with input dataframe."""
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    def prepare_data(self, city=None, target='arbovirus_bool'):
        """Prepare data for forecasting."""
        # If no city specified, aggregate all cities
        if city:
            data = self.df[self.df['city'] == city].copy()
        else:
            data = self.df.copy()
            
        # Aggregate daily cases
        daily_cases = data.groupby('date')[target].sum().reset_index()
        
        # Prepare for Prophet
        prophet_df = daily_cases.rename(columns={
            'date': 'ds',
            target: 'y'
        })
        
        return prophet_df
    
    def train_and_forecast(self, df, forecast_periods=90):
        """Train Prophet model and generate forecast."""
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        return forecast
    
    def plot_forecast(self, city=None, forecast_periods=90):
        """Create simple visualization of forecast."""
        # Prepare data
        prophet_df = self.prepare_data(city)
        
        # Generate forecast
        forecast = self.train_and_forecast(prophet_df, forecast_periods)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Plot actual cases
        plt.plot(prophet_df['ds'], prophet_df['y'], 
                'k.', alpha=0.6, label='Actual Cases')
        
        # Plot forecast
        plt.plot(forecast['ds'], forecast['yhat'], 
                'b-', linewidth=2, label='Forecast')
        
        # Plot confidence interval
        plt.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'],
                        color='blue', alpha=0.2, 
                        label='95% Confidence Interval')
        
        # Styling
        title = f'Disease Cases Forecast{" for " + city if city else ""}'
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date')
        plt.ylabel('Number of Cases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
