import pandas as pd
import numpy as np
import os
from pathlib import Path

# Define the columns of interest
columns_of_interest = [
    'test_accuracy_1ticket',
    'test_accuracy_2ticket',
    'test_tpr_1ticket',
    'test_tpr_2ticket',
    'test_disparity_1ticket',
    'test_disparity_2_ticket'  # Note: the column name has underscore before "ticket"
]

# Sample size
n = 500

# Z-score for 95% confidence interval
z_95 = 1.96

# Get all CSV files in the directory
directory = Path(__file__).parent
csv_files = sorted(directory.glob('results_*.csv'))

# Store results
results = []

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file.name}")
    print("=" * 80)
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract mean and std
    # The CSV has: header (row 0), count (row 0 in df), mean (row 1 in df), std (row 2 in df)
    mean_row = df.iloc[1]  # mean row
    std_row = df.iloc[2]   # std row
    
    file_results = {
        'file': csv_file.name
    }
    
    for col in columns_of_interest:
        if col in df.columns:
            mean_val = mean_row[col]
            std_val = std_row[col]
            
            # Calculate Standard Error
            se = std_val / np.sqrt(n)
            
            # Calculate 95% CI margin of error
            margin_error = z_95 * se
            
            
            file_results[f'{col}_mean'] = mean_val
            file_results[f'{col}_std'] = std_val
            file_results[f'{col}_se'] = se
            
            print(f"\n{col}:")
            print(f"  Mean: {mean_val:.6f}")
            print(f"  Std:  {std_val:.6f}")
            print(f"  SE:   {se:.6f}")
        else:
            print(f"Warning: Column '{col}' not found in {csv_file.name}")
    
    results.append(file_results)

# Create a summary DataFrame
summary_data = []
for result in results:
    row = {'file': result['file']}
    for col in columns_of_interest:
        if f'{col}_mean' in result:
            row[f'{col}_mean'] = result[f'{col}_mean']
            row[f'{col}_std'] = result[f'{col}_std']
            row[f'{col}_se'] = result[f'{col}_se']
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)

# Save to CSV
output_file = directory / 'summary_statistics.csv'
summary_df.to_csv(output_file, index=False)
print(f"\n\nSummary saved to: {output_file}")

# Display summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(summary_df.to_string(index=False))
