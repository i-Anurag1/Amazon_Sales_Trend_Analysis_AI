import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file missing: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        return self.df

    def create_features(self):
        print("Engineering Features")
        cat_col = 'Category' if 'Category' in self.df.columns else 'Item Type'
        
        all_categories = []
        unique_cats = self.df[cat_col].unique()
        
        for cat in unique_cats:
            #Filter & Resample
            cat_df = self.df[self.df[cat_col] == cat].copy()
            weekly_cat = cat_df.set_index('Date').resample('W')['Amount'].sum().reset_index()
            weekly_cat[cat_col] = cat
            
            weekly_cat['Lag_1'] = weekly_cat['Amount'].shift(1)
            weekly_cat['Lag_4'] = weekly_cat['Amount'].shift(4) 
            
            #Growth Rate
            weekly_cat['Growth_Rate'] = weekly_cat['Lag_1'].pct_change()
            weekly_cat['Growth_Rate'] = weekly_cat['Growth_Rate'].replace([np.inf, -np.inf], 0)
            weekly_cat['Growth_Rate'] = weekly_cat['Growth_Rate'].fillna(0)
            
            #Rolling Volatility
            weekly_cat['Roll_Std_4'] = weekly_cat['Amount'].shift(1).rolling(4).std()

            #Classification Target
            median_sales = weekly_cat['Amount'].median()
            weekly_cat['Trend_Target'] = (weekly_cat['Amount'] > median_sales).astype(int)
            
            all_categories.append(weekly_cat)
            
        self.df = pd.concat(all_categories)
        
        #Global Time Features
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        
        #One-Hot Encoding
        dummies = pd.get_dummies(self.df[cat_col], prefix='Cat')
        self.df = pd.concat([self.df, dummies], axis=1)
        
        #Standardize name to 'Category' for dashboard
        if 'Category' not in self.df.columns:
            self.df['Category'] = self.df[cat_col]
            
        self.df.dropna(inplace=True)

        print(f"Data Ready...Rows: {len(self.df)}")
        return self.df

    def save_data(self):
        self.df.to_csv(self.output_path, index=False)
        print(f"Saved to: {self.output_path}")

if __name__ == "__main__":
    engineer = FeatureEngineer("data/processed/clean_amazon_sales.csv", "data/processed/weekly_forecast_data.csv")
    engineer.load_data()
    engineer.create_features()
    engineer.save_data()