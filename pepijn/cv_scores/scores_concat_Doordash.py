import pandas as pd
from pathlib import Path

base = Path("/Users/lieveeberson/Documents/python3/FACT")
models_path = base / "pepijn/cv_scores"
original_path = base / "pepijn/original_cvs"
manipulate_path = base / "llm-hiring-ecosystem/Table1_Experimental_Modified_Resumes/Scores"

original_scores = pd.read_csv(models_path/"original_cvs.csv")
gemini_flash = pd.read_csv(models_path/"gemini_flash_doordash_pm.csv")
gemini_pro = pd.read_csv(models_path/"gemini_pro_doordash_pm.csv")
haiku = pd.read_csv(models_path/"haiku_doordash_pm.csv")
opus = pd.read_csv(models_path/"opus_doordash_pm.csv")
sonnet = pd.read_csv(models_path / "sonnet_doordash_pm.csv")
haiku_opus = pd.read_csv(models_path / "twice_haiku_opus_doordash_pm.csv")
sonnet_opus = pd.read_csv(models_path / "twice_sonnet_opus_doordash_pm.csv")
twice_opus = pd.read_csv(models_path / "twice_opus_doordash_pm.csv")

assert len(gemini_flash) == len(gemini_pro) == len(haiku) == len(opus) == len(sonnet) == len(haiku_opus) == len(sonnet_opus) == len(twice_opus), \
    "Model CSV row counts do not match"

target = pd.concat(
    [original_scores[["CV_doordash_pm_Score"]],
     gemini_flash[["Modified_gemini-3-flash-preview_of_CV_Modelgemini-3-flash-preview_doordash_pm_Score"]], 
     gemini_pro[["Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_doordash_pm_Score"]], 
     haiku[["Modified_claude-haiku-4-5_of_CV_Modelclaude-haiku-4-5_doordash_pm_Score"]], 
     opus[["Modified_claude-opus-4-5_of_CV_Modelclaude-opus-4-5_doordash_pm_Score"]], 
     sonnet[["Modified_claude-sonnet-4-5_of_CV_Modelclaude-sonnet-4-5_doordash_pm_Score"]],
     haiku_opus[["Modified_claude-opus-4-5_of_Modified_claude-haiku-4-5_of_CV_Modelclaude-haiku-4-5_Modelclaude-opus-4-5_doordash_pm_Score"]],
     sonnet_opus[["Modified_claude-opus-4-5_of_Modified_claude-sonnet-4-5_of_CV_Modelclaude-sonnet-4-5_Modelclaude-opus-4-5_doordash_pm_Score"]],
     twice_opus[["Modified_claude-opus-4-5_of_Modified_claude-opus-4-5_of_CV_Modelclaude-opus-4-5_Modelclaude-opus-4-5_doordash_pm_Score"]]],
    axis=1
)

original = pd.read_csv(original_path / "original_cvs.csv")

assert len(target) == len(original), \
    "Target and original_cvs row counts do not match"

target["True Label"] = (original["Position Group"] == "Project Manager").astype(int)

manipulated = pd.read_csv(manipulate_path / "DoorDashPMScores.csv")

assert len(target) == len(manipulated), \
    "Target and manipulated_cvs row counts do not match"

target["Will Manipulate"] = manipulated["Will Manipulate"]

target.to_csv("DoorDashPM_newscores.csv", index=False)