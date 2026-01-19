import pandas as pd
from pathlib import Path

# Define paths
base = Path("/Users/lieveeberson/Documents/python3/FACT/llm-hiring-ecosystem")
src_dir = base / "Same_resumes_scoredagain"
original_cv_path = base / "Table1_Experimental_Modified_Resumes" / "Original_CV.csv"
target_format_path = base / "Figure1_100Samples" / "Scores" / "doordash_job_description_prev.csv"
output_path = src_dir / "doordash_job_description_prev_formatted.csv"

# Read Original_CV.csv to get True Label
print("Reading Original_CV.csv...")
original_cv = pd.read_csv(original_cv_path)
print(f"Original_CV shape: {original_cv.shape}")

# Get True Label column
true_labels = original_cv['True Label'].values

# Read target format to see column order
target_df = pd.read_csv(target_format_path, nrows=0)
target_columns = target_df.columns.tolist()
print(f"Target columns: {target_columns}")

# Map each file to its target column name matching doordash_job_description_prev.csv format
# Note: Target file has leading spaces in some column names, we'll match exactly
# We only map files that have unique column names in the target format
# The "Cleaned" versions (gpt4o_once, twicegpt4o) don't exist in target, so we'll skip them
column_map = {
    "ScoresDoorDash_Original_File_chat3_5once.csv": " GPT-35 Conversation-Improved CV DoorDash PM Score",
    "ScoresDoorDash_Original_File_chat4o_on_chat35.csv": " GPT-4o Conversation-Improved CV DoorDash PM Score",
    "ScoresDoorDash_Original_File_chat4o_on_chat4omini.csv": " Twice GPT-4o Conversation-Improved CV DoorDash PM Score",
    "ScoresDoorDash_Original_File_chat4omini_once.csv": "GPT-4o-mini Improved CVDoorDash Score",
    "ScoresDoorDash_Original_File_Original_CV_for_scorecv.csv": "Original CVDoorDash Score",
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
if score_dfs:
    combined_scores = pd.concat(score_dfs, axis=1)
    # Remove any duplicate columns (in case of duplicate mappings)
    combined_scores = combined_scores.loc[:, ~combined_scores.columns.duplicated()]
else:
    combined_scores = pd.DataFrame()

# Get number of rows
n_rows = len(combined_scores)

# Create index columns matching the target format
# Target format has: empty column, Unnamed: 0.1, Unnamed: 0, then score columns, then True Label, then GPT-4o-mini and Original
final_df = pd.DataFrame({
    'Unnamed: 0.1': range(n_rows),
    'Unnamed: 0': range(n_rows)
})

# Add empty first column (unnamed) - we'll add it as the first column
final_df.insert(0, '', '')

# Add score columns that we have
# We'll add them in the order they appear in target format
for col in target_columns:
    if col in ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'True Label', 'GPT-4o-mini Improved CVDoorDash Score', 'Original CVDoorDash Score']:
        continue  # Skip these, we'll add them separately
    if col in combined_scores.columns:
        # Get the values as a 1D array
        values = combined_scores[col].values
        if len(values.shape) > 1:
            values = values.flatten()
        final_df[col] = values

# Add True Label column (positioned after score columns in target format)
final_df['True Label'] = true_labels[:n_rows]

# Add GPT-4o-mini and Original at the end if they exist
if 'GPT-4o-mini Improved CVDoorDash Score' in combined_scores.columns:
    final_df['GPT-4o-mini Improved CVDoorDash Score'] = combined_scores['GPT-4o-mini Improved CVDoorDash Score'].values
if 'Original CVDoorDash Score' in combined_scores.columns:
    final_df['Original CVDoorDash Score'] = combined_scores['Original CVDoorDash Score'].values

# Reorder columns to match target format exactly
# Start with empty column and index columns
ordered_columns = ['', 'Unnamed: 0.1', 'Unnamed: 0']

# Add score columns in the order they appear in target (excluding index and special columns)
# Only include columns we actually have
for col in target_columns:
    if col in ['Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'True Label', 'GPT-4o-mini Improved CVDoorDash Score', 'Original CVDoorDash Score']:
        continue
    if col in final_df.columns and col not in ordered_columns:
        ordered_columns.append(col)

# Add True Label after score columns
if 'True Label' in final_df.columns:
    ordered_columns.append('True Label')

# Add GPT-4o-mini and Original at the end
if 'GPT-4o-mini Improved CVDoorDash Score' in final_df.columns:
    ordered_columns.append('GPT-4o-mini Improved CVDoorDash Score')
if 'Original CVDoorDash Score' in final_df.columns:
    ordered_columns.append('Original CVDoorDash Score')

# Reorder the dataframe
final_df = final_df[ordered_columns]

# Save the result
print(f"\nSaving to {output_path}...")
final_df.to_csv(output_path, index=False)
print(f"Done! Created file with shape: {final_df.shape}")
print(f"Columns: {list(final_df.columns)}")
print(f"\nFirst few rows:")
print(final_df.head())

