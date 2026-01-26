# Extensions

The experiments in this directory are mainly extensions to the original
paper (Cohen, 2025). We introduce a new scoring model and test additional
modification models (beyond GPT-4o), with emphasis on Gemini and Claude.

Below is a concise map of data files, including which ones were created by me
and which ones were created by a groupmate as follow-up analyses.

## Files I created (original experimental data)

### Modified resumes (`pepijn/modified_cvs/`)
- **Pattern**: `{maybe_number_of_modifications}_{model}_{jobdescription}.csv`
  - Examples: `sonnet_doordash_pm.csv`, `twice_opus_google_ux.csv`, `fourth_haiku_doordash_pm.csv`
- These are the original LLM-modified resumes I generated.
- Important note from `modified_cvs/README.md`: the job description was **not**
  used in the prompt. So `*_doordash_pm.csv` and `*_google_ux.csv` are just two
  **stochastic runs** of the same setup.

### Scores for those modified resumes (`pepijn/cv_scores/`)
- **Pattern**: same filename as above, but containing **scores** for *all* job
  descriptions. Use the column ending with `_{job}_Score`.
- Example: `sonnet_doordash_pm.csv` contains columns for `*_doordash_pm_Score`,
  `*_google_ux_Score`, etc.
- **Scorer**: these files are from the original scorer (ResumeMatcher), not the
  new MiniLM scorer.

## Groupmate follow-up artifacts (not made by me)

### New scorer datasets (MiniLM)
- `pepijn/cv_scores/DoorDashPM_newscores.csv`
- `pepijn/cv_scores/GoogleUX_newscores.csv`
- These are **new-scorer** results (MiniLM-based) and appear to be **single-job**
  bundles: each file only contains scores for one job (DoorDash PM or Google UX)
  across multiple model variants. They also include labels like `True Label` /
  `UX True Label` and `Will Manipulate`.

### Two-ticket scheme results (new scorer)
- `pepijn/cv_scores/final_results_new_scorer_claude/`
- `pepijn/cv_scores/final_results_new_scorer_gemini/`
- Contain `results_*.csv` and `summary_statistics.csv` for two-ticket scheme
  experiments under the new scorer. These are used to evaluate Claims 4–5.

### Exploratory plots
- `pepijn/cv_scores/opus_newscores_DD.png`
- `pepijn/cv_scores/opus_newscores_G.png`
- PNG plots produced during new-scorer analyses.

### Helper scripts and notebooks (new scorer + two-ticket)
- `pepijn/cv_scores/scores_concat_Doordash.py`
- `pepijn/cv_scores/scores_concat_Google.py`
- `pepijn/cv_scores/significance_tests_newmodified_newscorer.ipynb`
- `pepijn/cv_scores/significance_tests_newnew_gemini.ipynb`
- These are the scripts/notebooks used to compute and summarize the two-ticket
  results for the new scorer and Gemini/Claude runs.

If any of the above is incorrect, check the git history or ask the original
author. The visualization script (`pepijn/visualize.py`) uses these datasets
explicitly and labels them as groupmate follow-ups where appropriate.
