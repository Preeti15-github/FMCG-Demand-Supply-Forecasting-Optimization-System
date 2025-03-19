import os

file_path = r"C:\Users\Preeti\Downloads\FMCG PROJECT\extended_fmcg_demand_forecasting.csv"

if os.path.exists(file_path):
    print("✅ File found!")
else:
    print("❌ File not found! Check the path.")


import pandas as pd

df = pd.read_csv(file_path)
print("Column Names:", df.columns)
print(df.head())  # Preview the first few rows
