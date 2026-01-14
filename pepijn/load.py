import os
import yaml


def get_pm_job_descriptions(
    job_descriptions_folder="job_descriptions",
    job_descriptions_pm_folder_name="PM_job_descriptions",
) -> tuple[list[str], list[str]]:
    job_descriptions_pm_folder = os.path.join(
        job_descriptions_folder, job_descriptions_pm_folder_name
    )

    job_names_pm = []
    job_descriptions_pm = []

    for filename in os.listdir(job_descriptions_pm_folder):
        if filename.endswith(".txt"):
            job_names_pm.append(filename[:-4])  # Remove .txt extension
            with open(os.path.join(job_descriptions_pm_folder, filename), "r") as f:
                job_descriptions_pm.append(f.read().strip())

    return job_names_pm, job_descriptions_pm


def get_uiux_job_descriptions(
    job_descriptions_folder="job_descriptions",
    job_descriptions_uiux_folder_name="UX_job_descriptions",
) -> tuple[list[str], list[str]]:

    job_descriptions_uiux_folder = os.path.join(
        job_descriptions_folder, job_descriptions_uiux_folder_name
    )

    job_names_uiux = []
    job_descriptions_uiux = []

    for filename in os.listdir(job_descriptions_uiux_folder):
        if filename.endswith(".txt"):
            job_names_uiux.append(filename[:-4])  # Remove .txt extension
            with open(os.path.join(job_descriptions_uiux_folder, filename), "r") as f:
                job_descriptions_uiux.append(f.read().strip())

    return job_names_uiux, job_descriptions_uiux


def get_modification_prompts(
    modification_prompts_folder="modify_prompts",
) -> tuple[list[str], list[str]]:

    modification_prompts = []
    modification_prompt_names = []

    for filename in os.listdir(modification_prompts_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(modification_prompts_folder, filename), "r") as f:
                modification_prompts.append(f.read().strip())
            modification_prompt_names.append(filename[:-4])  # Remove .txt extension

    return modification_prompt_names, modification_prompts


def load_api_keys() -> dict[str, str]:
    with open("api_keys.yaml", "r") as f:
        api_keys = yaml.safe_load(f)
        api_keys = {
            provider: info["api_key"] for provider, info in api_keys["services"].items()
        }
    return api_keys
