import pandas as pd
from pathlib import Path

base = Path("/Users/lieveeberson/Documents/python3/FACT")
models_path = base / "llm-hiring-ecosystem/new_modified_scores/all_scores"
original_path = base / "llm-hiring-ecosystem/new_modified_scores"
manipulate_path = base / "llm-hiring-ecosystem/Table1_Experimental_Modified_Resumes/Scores"

original_scores = pd.read_csv(models_path/"ScoresUIUX_Original_File_merged.csv")
gpt4o = pd.read_csv(models_path/"ScoresUIUX_Original_File_merged_GPT-4o1.csv")
gpt35 = pd.read_csv(models_path/"ScoresUIUX_Original_File_merged_GPT-3.5.1.csv")
gptmini = pd.read_csv(models_path/"ScoresUIUX_Original_File_merged_GPT-mini1.csv")
twice_4o = pd.read_csv(models_path/"ScoresUIUX_Original_File_merged_GPT-4o2.csv")
first35_4o = pd.read_csv(models_path/"ScoresUIUX_Original_File_merged_gpt4o-on3.5.csv")
mini_4o = pd.read_csv(models_path/"ScoresUIUX_Original_File_merged_gpt4o-onmini.csv")

assert len(original_scores) == len(gpt4o) == len(gpt35) == len(gptmini) == len(twice_4o) == len(first35_4o) == len(mini_4o), \
    "Model CSV row counts do not match"

gpt4o = gpt4o.rename(columns={
    "Modified_openai/gpt-4o_of_resume_Modelopenai/gpt-4o UIUX Score": "gpt_4o"
})

gpt35 = gpt35.rename(columns={
    "Modified_openai/gpt-3.5-turbo_of_resume_Modelopenai/gpt-3.5-turbo UIUX Score": "gpt_3.5"
})

gptmini = gptmini.rename(columns={
    "Modified_openai/gpt-4o-mini_of_resume_Modelopenai/gpt-4o-mini UIUX Score": "gpt_mini"
})

twice_4o = twice_4o.rename(columns={
    "Modified_openai/gpt-4o_of_Modified_openai/gpt-4o_of_resume_Modelopenai/gpt-4o_Modelopenai/gpt-4o Google Score": "gpt4o_twice"
})

first35_4o = first35_4o.rename(columns={
    "Modifiedgpt4o-on3.5 UIUX Score": "4o_on_35"
})

mini_4o = mini_4o.rename(columns={
    "Modifiedgpt4o-onmini UIUX Score": "4o_on_mini"
})


target = pd.concat(
    [original_scores[["resume UIUX Score"]],
     gpt4o,
     gptmini,
     gpt35,
     twice_4o,
     first35_4o,
     mini_4o],
    axis=1
)

target["True Label"] = 0
target.loc[:259, "True Label"] = 1

manipulated = pd.read_csv(manipulate_path / "GoogleUX_Scores.csv")

assert len(target) == len(manipulated), \
    "Target and manipulated_cvs row counts do not match"

target["Will Manipulate"] = manipulated["Will Manipulate"]

target.to_csv("GoogleUX_newscores.csv", index=False)