import pandas as pd
from pathlib import Path

base = Path("/Users/lieveeberson/Documents/python3/FACT")
models_path = base / "pepijn/cv_scores"
original_path = base / "pepijn/original_cvs"
manipulate_path = base / "llm-hiring-ecosystem/Table1_Experimental_Modified_Resumes/Scores"

original_scores = pd.read_csv(models_path/"original_cvs.csv")
gemini_flash = pd.read_csv(models_path/"gemini_flash_google_ux.csv")
gemini_pro = pd.read_csv(models_path/"gemini_pro_google_ux.csv")
gemini_flash_lite = pd.read_csv(models_path/"gemini_flash_lite_google_ux.csv")
haiku = pd.read_csv(models_path/"haiku_google_ux.csv")
opus = pd.read_csv(models_path/"opus_google_ux.csv")
sonnet = pd.read_csv(models_path / "sonnet_google_ux.csv")
haiku_opus = pd.read_csv(models_path / "twice_haiku_opus_google_ux.csv")
sonnet_opus = pd.read_csv(models_path / "twice_sonnet_opus_google_ux.csv")
twice_opus = pd.read_csv(models_path / "twice_opus_google_ux.csv")
twice_gemini_flash = pd.read_csv(models_path / "twice_gemini_flash_google_ux.csv")
twice_gemini_pro = pd.read_csv(models_path / "twice_gemini_pro_google_ux.csv")
twice_gemini_flash_lite = pd.read_csv(models_path / "twice_gemini_flash_lite_google_ux.csv")
twice_gemini_flash_pro = pd.read_csv(models_path / "twice_gemini_flash_pro_google_ux.csv")
twice_gemini_flash_lite_pro = pd.read_csv(models_path / "twice_gemini_flash_lite_pro_google_ux.csv")

assert len(gemini_flash) == len(gemini_pro) == len(gemini_flash_lite) == len(haiku) == len(opus) == len(sonnet) == len(haiku_opus) == len(sonnet_opus) == len(twice_opus) == len(twice_gemini_flash) == len(twice_gemini_pro) == len(twice_gemini_flash_lite) == len(twice_gemini_flash_pro) == len(twice_gemini_flash_lite_pro), \
    "Model CSV row counts do not match"

target = pd.concat(
    [original_scores[["CV_google_ux_Score"]],
     gemini_flash[["Modified_gemini-3-flash-preview_of_CV_Modelgemini-3-flash-preview_google_ux_Score"]].rename(columns={"Modified_gemini-3-flash-preview_of_CV_Modelgemini-3-flash-preview_google_ux_Score": "gemini_flash_google_ux_Score"}), 
     gemini_pro[["Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score"]].rename(columns={"Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score": "gemini_pro_google_ux_Score"}), 
     gemini_flash_lite[["Modified_gemini-2.0-flash-lite_of_CV_Modelgemini-2.0-flash-lite_google_ux_Score"]].rename(columns={"Modified_gemini-2.0-flash-lite_of_CV_Modelgemini-2.0-flash-lite_google_ux_Score": "gemini_flash_lite_google_ux_Score"}),
     haiku[["Modified_claude-haiku-4-5_of_CV_Modelclaude-haiku-4-5_google_ux_Score"]], 
     opus[["Modified_claude-opus-4-5_of_CV_Modelclaude-opus-4-5_google_ux_Score"]], 
     sonnet[["Modified_claude-sonnet-4-5_of_CV_Modelclaude-sonnet-4-5_google_ux_Score"]],
     haiku_opus[["Modified_claude-opus-4-5_of_Modified_claude-haiku-4-5_of_CV_Modelclaude-haiku-4-5_Modelclaude-opus-4-5_google_ux_Score"]],
     sonnet_opus[["Modified_claude-opus-4-5_of_Modified_claude-sonnet-4-5_of_CV_Modelclaude-sonnet-4-5_Modelclaude-opus-4-5_google_ux_Score"]],
     twice_opus[["Modified_claude-opus-4-5_of_Modified_claude-opus-4-5_of_CV_Modelclaude-opus-4-5_Modelclaude-opus-4-5_google_ux_Score"]],
     twice_gemini_flash[["Modified_gemini-3-flash-preview_of_CV_Modelgemini-3-flash-preview_google_ux_Score"]].rename(columns={"Modified_gemini-3-flash-preview_of_CV_Modelgemini-3-flash-preview_google_ux_Score": "twice_gemini_flash_google_ux_Score"}),
     twice_gemini_pro[["Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score"]].rename(columns={"Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score": "twice_gemini_pro_google_ux_Score"}),
     twice_gemini_flash_lite[["Modified_gemini-2.0-flash-lite_of_CV_Modelgemini-2.0-flash-lite_google_ux_Score"]].rename(columns={"Modified_gemini-2.0-flash-lite_of_CV_Modelgemini-2.0-flash-lite_google_ux_Score": "twice_gemini_flash_lite_google_ux_Score"}),
     twice_gemini_flash_pro[["Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score"]].rename(columns={"Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score": "twice_gemini_flash_pro_google_ux_Score"}),
     twice_gemini_flash_lite_pro[["Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score"]].rename(columns={"Modified_gemini-3-pro-preview_of_CV_Modelgemini-3-pro-preview_google_ux_Score": "twice_gemini_flash_lite_pro_google_ux_Score"})],
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
