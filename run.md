# What TODO to get this running.

## Non-existent index column in score.py

For making score.py work, I had to change

```input_csv = pd.read_csv(str(resume_path), index_col="id")```

to 

```input_csv = pd.read_csv(str(resume_path), index_col=0)```.

because there is no "id" column in the sample input csv files.

## Api keys file format

Also, the expected format for the api keys yaml file is:

```yaml

services:
  openai:
    api_key: your_openai_api_key_here
  anthropic:
    api_key: your_anthropic_api_key_here

```

## Outdated model in modify_cv.py

Lastly the model they used in modify_cv.py is outdated:

```python
self.model = model if model else "claude-3-sonnet-20240229" # doesn't exist anymore
```

Current activate models are listed here: https://docs.anthropic.com/claude/reference/models

For instance we could use `claude-sonnet-4-5-20250929` or `claude-opus-4-5-20251101`
or `claude-haiku-4-5-20251001` or without the tag at the end to always get the newest.

## Embedding similarity in final_figures.ipynb

Somewhere in the notebook, you'll get an error when it's trying to find some embedding tables in Figure1_100Samples/Embeddings that doesnt exist. Probably to big to store on github. There seems to be no code for generating these embeddings, but in the paper they mention: 

"Figure 8. Distribution of cosine similarities (Sentence Embeddings ALL-MINILM-L6-V2) across all pairs of resumes. Compare to no manipulation, nearly all models increase the similarity of resumes, especially CLAUDE-3.5-SONNET."

So we gotta use this Sentence Embeddings ALL-MINILM-L6-V2 model I guess. 

you can use it from huggingface:

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

But this is just a small appendix thing which is only references once in the paper, so maybe not super important.

## `doordash_df` in final_figures.ipynb

They reference some `doordash_df` dataframe that is not defined anywhere. Is this just `melted_df`? Don't think so, still getting key errors after changing that.

## Wrong filenames in consistency_checks.ipynb and significance_tests.ipynb

In these notebooks they look for `../Figure1_100Samples/Scores/doordash_job_description.csv` but the correct filename is `doordash_job_description_prev.csv`. Same for the google ux file.

## Data fiksen

De hele dataset kan je vanuit de `recruitment-dataset` repo halen met DVC. Zie de instructies in de `README.md` van die repo. Hiervoor moet je wel je python versie van 3.10 (uit die `llm-hiring-ecosystem/env.yml` file) naar 3.11 upgraden, want DVC ondersteunt geen 3.10 meer.




