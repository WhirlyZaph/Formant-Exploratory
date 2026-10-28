import pandas as pd

# Load your CSV
df = pd.read_csv("1fake.csv")

# Convert F5 to numeric
df['F5(Hz)'] = pd.to_numeric(df['F5(Hz)'], errors='coerce')

# Check high-frequency F5 values
high_f5 = df[df['F5(Hz)'] > 5000]['F5(Hz)']
print(f"Number of F5 values above 5000 Hz: {len(high_f5)}")
print(f"Mean: {high_f5.mean():.0f} Hz")
print(f"Values: {high_f5.values[:10]}")  # Show first 10