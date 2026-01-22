import pandas as pd
from pathlib import Path

base = Path("/Users/lieveeberson/Documents/python3/FACT")
models_path = base / "llm-hiring-ecosystem/UK_data/all_scores"
original_path = base / "llm-hiring-ecosystem/UK_data"
manipulate_path = base / "llm-hiring-ecosystem/Table1_Experimental_Modified_Resumes/Scores"

original_scores = pd.read_csv(models_path/"ScoresDoordash_Original_File_uiux_pm_520.csv")
gpt4o = pd.read_csv(models_path/"ScoresDoordash_Original_File_merged_gpt-4o_UK.csv")
gpt35 = pd.read_csv(models_path/"ScoresDoordash_Original_File_merged_GPT-3.5_UK.csv")
gptmini = pd.read_csv(models_path/"ScoresDoordash_Original_File_merged_GPT-mini_UK.csv")
twice_4o = pd.read_csv(models_path/"ScoresDoordash_Original_File_merged_gpt-4o2_UK.csv")
first35_4o = pd.read_csv(models_path/"ScoresDoordash_Original_File_merged_gpt4o-on3.5_UK.csv")
mini_4o = pd.read_csv(models_path/"ScoresDoordash_Original_File_merged_gpt4o_onmini_UK.csv")

assert len(original_scores) == len(gpt4o) == len(gpt35) == len(gptmini) == len(twice_4o) == len(first35_4o) == len(mini_4o), \
    "Model CSV row counts do not match"

gpt4o = gpt4o.rename(columns={
    "Modified_openai/gpt-4o_of_resume_Modelopenai/gpt-4o DoorDash Score": "gpt_4o"
})

gpt35 = gpt35.rename(columns={
    "Modified_openai/gpt-3.5-turbo_of_resume_Modelopenai/gpt-3.5-turbo DoorDash Score": "gpt_3.5"
})

gptmini = gptmini.rename(columns={
    "Modified_openai/gpt-4o-mini_of_resume_Modelopenai/gpt-4o-mini DoorDash Score": "gpt_mini"
})

twice_4o = twice_4o.rename(columns={
    "Modified_openai/gpt-4o_of_Modified_openai/gpt-4o_of_resume_Modelopenai/gpt-4o_Modelopenai/gpt-4o DoorDash Score": "gpt4o_twice"
})

first35_4o = first35_4o.rename(columns={
    "Modified_openai/gpt-4o_of_Modified_openai/gpt-3.5-turbo_of_resume_Modelopenai/gpt-3.5-turbo_Modelopenai/gpt-4o DoorDash Score": "4o_on_35"
})

mini_4o = mini_4o.rename(columns={
    "Modified_openai/gpt-4o_of_Modified_openai/gpt-4o-mini_of_resume_Modelopenai/gpt-4o-mini_Modelopenai/gpt-4o DoorDash Score": "4o_on_mini"
})

target = pd.concat(
    [original_scores[["CV DoorDash Score"]],
     gpt4o,
     gptmini,
     gpt35,
     twice_4o,
     first35_4o,
     mini_4o],
    axis=1
)


original = pd.read_csv(original_path / "uiux_pm_combined_520.csv")

assert len(target) == len(original), \
    "Target and original_cvs row counts do not match"

target["True Label"] = original["Position"].str.contains("Project Manager", na=False).astype(int)

manipulated = pd.read_csv(manipulate_path / "DoorDashPMScores.csv")

assert len(target) == len(manipulated), \
    "Target and manipulated_cvs row counts do not match"

target["Will Manipulate"] = manipulated["Will Manipulate"]

target.to_csv("DoorDashPM_newscores.csv", index=False)