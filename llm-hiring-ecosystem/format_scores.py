import pandas as pd
from pathlib import Path

# Define paths
base = Path("/Users/lieveeberson/Documents/python3/FACT/llm-hiring-ecosystem")
src_dir = base / "Same_resumes_scoredagain"
original_cv_path = base / "Table1_Experimental_Modified_Resumes" / "Original_CV.csv"
target_format_path = base / "Table1_Experimental_Modified_Resumes" / "Scores" / "DoorDashPMScores.csv"
output_path = src_dir / "DoorDashPMScores_formatted.csv"

# Read target file early to get True Label and Will Manipulate columns
print("Reading target file to get True Label and Will Manipulate...")
target_full = pd.read_csv(target_format_path)
print(f"Target file shape: {target_full.shape}")

# Read Original_CV.csv to get Position and id
print("Reading Original_CV.csv...")
original_cv = pd.read_csv(original_cv_path)
print(f"Original_CV shape: {original_cv.shape}")

# Get columns from Original_CV.csv
positions = original_cv['Position'].values
ids = original_cv['id'].values

# Map each file to its target column name matching DoorDashPMScores.csv format
column_map = {
    "ScoresDoorDash_Original_File_Original_CV_for_scorecv.csv": "CVDoorDash PM Score",
    "ScoresDoorDash_Original_File_gpt4o_once.csv": "Cleaned GPT-4o Conversation-Improved CVDoorDash PM Score",
    "ScoresDoorDash_Original_File_twicegpt4o.csv": "Cleaned Twice GPT-4o Conversation-Improved CVDoorDash PM Score",
    "ScoresDoorDash_Original_File_chat3_5once.csv": "GPT-3.5 Improved CVDoorDash PM Score",
    "ScoresDoorDash_Original_File_chat4o_on_chat35.csv": "GPT-4o Conversation Improved on GPT-3.5 Improved CVDoorDash PM Score",
    "ScoresDoorDash_Original_File_chat4omini_once.csv": "GPT-4o-mini Improved CVDoorDash PM Score",
    "ScoresDoorDash_Original_File_chat4o_on_chat4omini.csv": "GPT-4o Conversation Improved on GPT-4o-mini Improved CVDoorDash PM Score",
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

# Create the final dataframe with Position, id first
final_df = pd.DataFrame({
    'Position': positions[:n_rows],
    'id': ids[:n_rows]
})

# Add score columns in the order they appear in target format
for col in target_columns:
    if col in ['Position', 'True Label', 'id', 'Will Manipulate']:
        continue  # Skip these, we'll add them separately
    if col in combined_scores.columns:
        final_df[col] = combined_scores[col].values
    else:
        print(f"  Warning: Column '{col}' not found in combined scores")

# Copy True Label and Will Manipulate columns directly from target file
# Assuming rows are in the same order
print("\nCopying True Label and Will Manipulate from target file...")
if n_rows <= len(target_full):
    final_df['True Label'] = target_full['True Label'].values[:n_rows]
    final_df['Will Manipulate'] = target_full['Will Manipulate'].values[:n_rows]
    print(f"Copied {n_rows} rows of True Label and Will Manipulate")
    print(f"True Label distribution: {final_df['True Label'].value_counts().to_dict()}")
    print(f"Will Manipulate distribution: {final_df['Will Manipulate'].value_counts().to_dict()}")
else:
    print(f"Warning: Output has {n_rows} rows but target has {len(target_full)} rows")
    # Copy what we can
    final_df['True Label'] = target_full['True Label'].values
    final_df['Will Manipulate'] = target_full['Will Manipulate'].values
    # Fill remaining with NaN or default values
    final_df.loc[len(target_full):, 'True Label'] = pd.NA
    final_df.loc[len(target_full):, 'Will Manipulate'] = False

# Reorder columns to match target format exactly
ordered_columns = [col for col in target_columns if col in final_df.columns]
# Add any extra columns that might be in final_df but not in target
for col in final_df.columns:
    if col not in ordered_columns:
        ordered_columns.append(col)

final_df = final_df[ordered_columns]

# Save the result
print(f"\nSaving to {output_path}...")
final_df.to_csv(output_path, index=False)
print(f"Done! Created file with shape: {final_df.shape}")
print(f"Columns: {list(final_df.columns)}")
print(f"\nFirst few rows:")
print(final_df.head())

