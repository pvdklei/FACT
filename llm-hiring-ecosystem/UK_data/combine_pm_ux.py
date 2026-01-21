import pandas as pd

# Load CSV files
uiux_df = pd.read_csv("uiux_first_260.csv")
pm_df = pd.read_csv("pm_first_260.csv")

# Concatenate in order: UI/UX first, PM second
combined_df = pd.concat([uiux_df, pm_df], ignore_index=True)

# Save to new CSV
combined_df.to_csv("uiux_pm_combined_520.csv", index=False)