import datasets
from datasets import load_dataset
import os
from pathlib import Path

candidateprofiles = load_dataset("lang-uk/recruitment-dataset-candidate-profiles-ukrainian")
jobdescriptions = load_dataset("lang-uk/recruitment-dataset-job-descriptions-ukrainian")

jobdescriptions_df = jobdescriptions["train"].to_pandas()
candidateprofiles_df = candidateprofiles["train"].to_pandas()

uiux_df = candidateprofiles_df[candidateprofiles_df["Position"].str.contains("UI/UX Designer", na=False)]
pm_df = candidateprofiles_df[candidateprofiles_df["Position"].str.contains("Project Manager", na=False)]

uiux_first_260_df = uiux_df.head(260)
pm_first_260_df = pm_df.head(260)

uiux_first_260_df.to_csv("uiux_first_260.csv", index=False)
pm_first_260_df.to_csv("pm_first_260.csv", index=False)