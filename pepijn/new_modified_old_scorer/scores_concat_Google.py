import pandas as pd
from pathlib import Path

base = Path("/Users/lieveeberson/Documents/python3/FACT")
models_path = base / "pepijn/cv_scores"
original_path = base / "pepijn/original_cvs"
manipulate_path = base / "llm-hiring-ecosystem/Table1_Experimental_Modified_Resumes/Scores"

gemini_flash = pd.read_csv(models_path/"gemini_flash_google_ux.csv")
gemini_pro = pd.read_csv(models_path/"gemini_pro_google_ux.csv")
haiku = pd.read_csv(models_path/"haiku_google_ux.csv")
opus = pd.read_csv(models_path/"opus_google_ux.csv")
sonnet = pd.read_csv(models_path / "sonnet_google_ux.csv")

assert len(gemini_flash) == len(gemini_pro) == len(haiku) == len(opus) == len(sonnet), \
    "Model CSV row counts do not match"

target = pd.concat(
    [gemini_flash[["Modified_gemini-3-flash-preview_of_CV_Modelgemini-3-flash-preview_google_ux_Score"]], gemini_pro[["Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score"]], haiku[["Modified_claude-haiku-4-5_of_CV_Modelclaude-haiku-4-5_google_ux_Score"]], opus[["Modified_claude-opus-4-5_of_CV_Modelclaude-opus-4-5_google_ux_Score"]], sonnet[["Modified_claude-sonnet-4-5_of_CV_Modelclaude-sonnet-4-5_google_ux_Score"]]],
    axis=1
)

original = pd.read_csv(original_path / "original_cvs.csv")

assert len(target) == len(original), \
    "Target and original_cvs row counts do not match"

target["UX True Label"] = (original["Position Group"] == "UI/UX Designer").astype(int)

manipulated = pd.read_csv(manipulate_path / "GoogleUX_Scores.csv")

assert len(target) == len(manipulated), \
    "Target and manipulated_cvs row counts do not match"

target["Will Manipulate"] = manipulated["Will Manipulate"]

target.to_csv("GoogleUX_newscores.csv", index=False)
