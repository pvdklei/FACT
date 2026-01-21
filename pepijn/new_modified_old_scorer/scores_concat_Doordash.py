import pandas as pd
from pathlib import Path

base = Path("/Users/lieveeberson/Documents/python3/FACT")
models_path = base / "pepijn/new_modified_old_scorer"
original_path = base / "pepijn/original_cvs"
manipulate_path = base / "llm-hiring-ecosystem/Table1_Experimental_Modified_Resumes/Scores"

original_scores = pd.read_csv(models_path/"ScoresDoorDash_Original_File_original_cvs_for_scorecv.csv")
gemini_flash = pd.read_csv(models_path/"ScoresDoorDash_Original_File_gemini_flash_v1.csv")
gemini_pro = pd.read_csv(models_path/"ScoresDoorDash_Original_File_gemini_pro_v1.csv")
gemini_flashlite = pd.read_csv(models_path/"ScoresDoorDash_Original_File_gemini_flash_lite_v1.csv")
gemini_pro_twice = pd.read_csv(models_path/"ScoresDoorDash_Original_File_twice_gemini_pro_v1.csv")
flash_pro = pd.read_csv(models_path/"ScoresDoorDash_Original_File_twice_gemini_flash_pro_v1.csv")
flashlite_pro = pd.read_csv(models_path/"ScoresDoorDash_Original_File_twice_gemini_flash_lite_pro_v1.csv")
haiku = pd.read_csv(models_path/"ScoresDoorDash_Original_File_haiku_v1.csv")
opus = pd.read_csv(models_path/"ScoresDoorDash_Original_File_opus_v1.csv")
sonnet = pd.read_csv(models_path / "ScoresDoorDash_Original_File_sonnet_v1.csv")
haiku_opus = pd.read_csv(models_path/"ScoresDoorDash_Original_File_twice_haiku_opus_v1.csv")
sonnet_opus = pd.read_csv(models_path/"ScoresDoorDash_Original_File_twice_sonnet_opus_v1.csv")
twice_opus = pd.read_csv(models_path/"ScoresDoorDash_Original_File_twice_opus_v1.csv")

assert len(gemini_flash) == len(gemini_pro) == len(haiku) == len(opus) == len(sonnet) == len(haiku_opus) == len(sonnet_opus) == len(twice_opus), \
    "Model CSV row counts do not match"

gemini_flash = gemini_flash.rename(columns={
    "Modified_gemini-3-flash-preview_of_CV_Modelgemini-3-flash-preview DoorDash Score": "gemini_flash"
})

gemini_pro = gemini_pro.rename(columns={
    "Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview DoorDash Score": "gemini_pro"
})

gemini_flashlite = gemini_flashlite.rename(columns={
    "Modified_gemini-2.0-flash-lite_of_CV_Modelgemini-2.0-flash-lite DoorDash Score": "gemini_flashlite"
})

gemini_pro_twice = gemini_pro_twice.rename(columns={
    "Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview DoorDash Score": "gemini_pro_twice"
})

flash_pro = flash_pro.rename(columns={
    "Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview DoorDash Score": "flash_pro"
})

flashlite_pro = flashlite_pro.rename(columns={
    "Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview DoorDash Score": "flashlite_pro"
})

target = pd.concat(
    [original_scores[["CV DoorDash Score"]],
     gemini_flash,
     gemini_pro,
     gemini_flashlite,
     gemini_pro_twice,
     flash_pro,
     flashlite_pro,
     haiku[["Modified_claude-haiku-4-5_of_CV_Modelclaude-haiku-4-5 DoorDash Score"]], 
     opus[["Modified_claude-opus-4-5_of_CV_Modelclaude-opus-4-5 DoorDash Score"]], 
     sonnet[["Modified_claude-sonnet-4-5_of_CV_Modelclaude-sonnet-4-5 DoorDash Score"]],
     haiku_opus[["Modified_claude-opus-4-5_of_Modified_claude-haiku-4-5_of_CV_Modelclaude-haiku-4-5_Modelclaude-opus-4-5 DoorDash Score"]],
     sonnet_opus[["Modified_claude-opus-4-5_of_Modified_claude-sonnet-4-5_of_CV_Modelclaude-sonnet-4-5_Modelclaude-opus-4-5 DoorDash Score"]],
     twice_opus[["Modified_claude-opus-4-5_of_Modified_claude-opus-4-5_of_CV_Modelclaude-opus-4-5_Modelclaude-opus-4-5 DoorDash Score"]]],
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