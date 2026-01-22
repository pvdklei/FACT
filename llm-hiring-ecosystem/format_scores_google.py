import pandas as pd
from pathlib import Path

# Define paths
base = Path("/Users/lieveeberson/Documents/python3/FACT/llm-hiring-ecosystem")
src_dir = base / "Same_resumes_scoredagain"
original_cv_path = base / "Table1_Experimental_Modified_Resumes" / "Original_CV.csv"
target_format_path = base / "Table1_Experimental_Modified_Resumes" / "Scores" / "GoogleUX_Scores.csv"
output_path = src_dir / "GoogleUX_scores_formatted.csv"

# Read target file early to get UX True Label and Will Manipulate columns
print("Reading target file to get UX True Label and Will Manipulate...")
target_full = pd.read_csv(target_format_path)
print(f"Target file shape: {target_full.shape}")

# Map each file to its target column name matching GoogleUX_Scores.csv format
column_map = {
    "ScoresGoogle_Original_File_twicegpt4o.csv": "Twice GPT-4o Google_UX Score",
    "ScoresGoogle_Original_File_gpt4o_once.csv": "Cleaned GPT-4o Conversation-Improved CVGoogle_UX Score",
    "ScoresGoogle_Original_File_chat3_5once.csv": "Modified GPT-3.5 Google_UX Score",
    "ScoresGoogle_Original_File_Original_CV_for_scorecv.csv": "CVGoogle_UX Score",
    "ScoresGoogle_Original_File_chat4o_on_chat35.csv": "Modified GPT-4o of GPT-3.5-turbo Google_UX Score",
    "ScoresGoogle_Original_File_chat4o_on_chat4omini.csv": "Modified GPT-4o of GPT-4o-mini Google_UX Score",
    "ScoresGoogle_Original_File_chat4omini_once.csv": "Modified GPT-4o-mini Google_UX Score",
}

# Read and combine all score files
score_dfs = []
for fname, target_col in column_map.items():
    file_path = src_dir / fname
    if file_path.exists():
        print(f"Reading {fname}...")
        df = pd.read_csv(file_path)
        
        # Get the score column (usually the second column, or first if only one)
        if len(df.columns) > 1:
            score_col = df.columns[1]
        else:
            score_col = df.columns[0]
        
        # Extract just the score values
        scores = df[score_col].values
        score_dfs.append(pd.DataFrame({target_col: scores}))
        print(f"  Extracted {len(scores)} scores")
    else:
        print(f"  Warning: {fname} not found, skipping")

# Combine all score columns
print("\nCombining scores...")
combined_scores = pd.concat(score_dfs, axis=1)

# Get number of rows
n_rows = len(combined_scores)

# Read target format to see column order
target_df = pd.read_csv(target_format_path, nrows=0)
target_columns = target_df.columns.tolist()
print(f"Target columns: {target_columns}")

# Create the final dataframe with empty first column
# Target format has: empty column, then score columns, then UX True Label, then Will Manipulate
final_df = pd.DataFrame(index=range(n_rows))
final_df.insert(0, '', '')  # Add empty first column

# Add score columns in the order they appear in target format
for col in target_columns:
    if col in ['', 'UX True Label', 'Will Manipulate', 'Unnamed: 0']:
        continue  # Skip these, we'll add them separately
    if col in combined_scores.columns:
        final_df[col] = combined_scores[col].values
    else:
        print(f"  Warning: Column '{col}' not found in combined scores")

# Copy UX True Label and Will Manipulate columns directly from target file
# Assuming rows are in the same order
print("\nCopying UX True Label and Will Manipulate from target file...")
if n_rows <= len(target_full):
    final_df['UX True Label'] = target_full['UX True Label'].values[:n_rows]
    final_df['Will Manipulate'] = target_full['Will Manipulate'].values[:n_rows]
    print(f"Copied {n_rows} rows of UX True Label and Will Manipulate")
    print(f"UX True Label distribution: {final_df['UX True Label'].value_counts().to_dict()}")
    print(f"Will Manipulate distribution: {final_df['Will Manipulate'].value_counts().to_dict()}")
else:
    print(f"Warning: Output has {n_rows} rows but target has {len(target_full)} rows")
    # Copy what we can
    final_df['UX True Label'] = target_full['UX True Label'].values
    final_df['Will Manipulate'] = target_full['Will Manipulate'].values
    # Fill remaining with NaN or default values
    final_df.loc[len(target_full):, 'UX True Label'] = pd.NA
    final_df.loc[len(target_full):, 'Will Manipulate'] = False

# Reorder columns to match target format exactly
ordered_columns = ['']  # Start with empty column

# Add score columns in target order
for col in target_columns:
    if col in ['', 'UX True Label', 'Will Manipulate', 'Unnamed: 0']:
        continue
    if col in final_df.columns and col not in ordered_columns:
        ordered_columns.append(col)

# Add UX True Label and Will Manipulate at the end
if 'UX True Label' in final_df.columns:
    ordered_columns.append('UX True Label')
if 'Will Manipulate' in final_df.columns:
    ordered_columns.append('Will Manipulate')

final_df = final_df[ordered_columns]

# Save the result
print(f"\nSaving to {output_path}...")
final_df.to_csv(output_path, index=False)
print(f"Done! Created file with shape: {final_df.shape}")
print(f"Columns: {list(final_df.columns)}")
print(f"\nFirst few rows:")
print(final_df.head())

