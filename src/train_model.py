import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

class HybridTrainer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.features = [c for c in self.df.columns if c not in ['Date', 'Amount', 'Trend_Target', 'Category', 'Item Type']]
        
    def train_price_model(self):
        print("\nTraining Price Model(Aggressive Random Forest)")
        X = self.df[self.features]
        y = self.df['Amount']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        
        model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=None, 
            random_state=42, 
            n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        print(f" Average Price Error: ${mae:,.2f}")
        joblib.dump(model, "models/price_model.pkl")

    def train_trend_model(self):
        print("\nTraining Trend Model(Classification)...")
        X = self.df[self.features]
        y = self.df['Trend_Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        
        model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=None, 
            random_state=42, 
            n_jobs=-1)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"Trend Accuracy Score: {acc*100:.2f}%")
        joblib.dump(model, "models/trend_model.pkl")

if __name__ == "__main__":
    if not os.path.exists("models"): os.makedirs("models")
    
    trainer = HybridTrainer("data/processed/weekly_forecast_data.csv")
    trainer.train_price_model()
    trainer.train_trend_model()