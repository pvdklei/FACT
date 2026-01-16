import pandas as pd
from pathlib import Path

base = Path("/Users/lieveeberson/Documents/python3/FACT/llm-hiring-ecosystem")
src = base / "Same_resumes_scoredagain"

# Map each file’s single column to the target column name you want
column_map = {
    "ScoresGoogle_Original_File_chat3_5once.csv": "GPT-35 Conversation-Improved CV Google PM Score",
    "ScoresGoogle_Original_File_chat4o_on_chat35.csv": "GPT-4o Conversation-Improved CV Google PM Score",
    "ScoresGoogle_Original_File_chat4o_on_chat4omini.csv": "Twice GPT-4o Conversation-Improved CV Google PM Score",
    "ScoresGoogle_Original_File_chat4omini_once.csv": "GPT-4o-mini Improved Google Score",
    "ScoresGoogle_Original_File_gpt4o_once.csv": "Cleaned GPT-4o Conversation-Improved CV Google Score",
    "ScoresGoogle_Original_File_twicegpt4o.csv": "Cleaned Twice GPT-4o Conversation-Improved CV Google Score",
    "ScoresGoogle_Original_File_Original_CV_for_scorecv.csv": "Original Google Score",
    # add more mappings if needed (e.g., project_manager_260)
}

dfs = []
for fname, target_col in column_map.items():
    df = pd.read_csv(src / fname)
    # first column is just an index; drop it and rename the score column
    score_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    df = df.rename(columns={score_col: target_col})
    df = df.drop(columns=[df.columns[0]], errors="ignore")  # drop unnamed/index
    dfs.append(df)

out = pd.concat(dfs, axis=1)

# Optional: reorder columns to mirror the prev file’s ordering
desired_order = pd.read_csv(base / "Figure1_100Samples/Scores/google_ux_job_description_prev.csv", nrows=0).columns.tolist()
# keep only those we have
out = out[[c for c in desired_order if c in out.columns] + [c for c in out.columns if c not in desired_order]]

out.to_csv(src / "google_ux_job_description_prev_formatted.csv", index=False)
print("Wrote:", src / "google_ux_job_description_prev_formatted.csv")