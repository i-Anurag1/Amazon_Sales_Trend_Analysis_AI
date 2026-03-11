import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Checks if file exists and loads it."""
    
        if not os.path.exists(self.file_path):
            print(f"Error: File not found at {self.file_path}")
            return None
        
        self.df = pd.read_csv(self.file_path)
        print(f"Data Loaded. Rows: {len(self.df)}")
        print(f"Columns found: {list(self.df.columns)}")
        return self.df

    def clean_data(self):
        if self.df is None:
            return None

        column_mapping = {
            'Order Date': 'Date', 
            'Total Revenue': 'Amount',
            'Order Priority': 'Status',
            'Item Type': 'Category'
        }
        self.df.rename(columns=column_mapping, inplace=True)

        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')

        print("Data Cleaned & Standardized.")
        return self.df

    def save_processed(self, output_path):
        """Saves the clean file for the next step."""
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    file_name = "Amazon Sales data.csv"
    input_path = os.path.join("data", "raw", file_name)
    output_path = os.path.join("data", "processed", "clean_amazon_sales.csv")

    loader = DataLoader(input_path)
    loader.load_data()
    loader.clean_data()
    loader.save_processed(output_path)