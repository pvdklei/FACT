# What TODO to get this running.

For making score.py work, I had to change

```input_csv = pd.read_csv(str(resume_path), index_col="id")```

to 

```input_csv = pd.read_csv(str(resume_path), index_col=0)```.

Also, the expected format for the api keys yaml file is:

```yaml

services:
  openai:
    api_key: your_openai_api_key_here
  anthropic:
    api_key: your_anthropic_api_key_here

```

Lastly the model they used in modify_cv.py is outdated:

```python
self.model = model if model else "claude-3-sonnet-20240229" # doesn't exist anymore
```

Current activate models are listed here: https://docs.anthropic.com/claude/reference/models

For instance we could use `claude-sonnet-4-5-20250929` or `claude-opus-4-5-20251101`
or `claude-haiku-4-5-20251001` or without the tag at the end to always get the newest.

