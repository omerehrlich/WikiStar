<div align="center">

<img width="500" alt="WikiSTAR_logo" src="https://github.com/user-attachments/assets/befecb22-4dfa-4b81-a94a-d51f33a3b326" />


# WikiSTAR

**Wikipedia Revision Classification Pipeline**

[![arXiv](https://img.shields.io/badge/arXiv-2607.12441-b31b1b.svg)](https://arxiv.org/abs/2607.12441)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![spaCy](https://img.shields.io/badge/spaCy-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![scispaCy](https://img.shields.io/badge/scispaCy-en__core__sci__md-0A9EDC)](https://allenai.github.io/scispacy/)
[![OpenAI](https://img.shields.io/badge/OpenAI%20Batch%20API-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/batch)
[![MediaWiki API](https://img.shields.io/badge/MediaWiki%20API-006699?logo=wikipedia&logoColor=white)](https://www.mediawiki.org/wiki/API:Main_page)

</div>

This repository contains the code for the paper
[**WikiSTAR: A System for Shedding Light on the Hidden History of Scientific
Wikipedia Articles**](https://arxiv.org/abs/2607.12441) (2026).

Classify the scientific content of Wikipedia article edits. Given an article
title, the pipeline fetches its full revision history, reconstructs how each
section changed over time, and asks a language model to label every
section-level edit against a taxonomy of scientific change types (new
information, references added, technical terms, and so on), with a short
explanation for each label.

## Pipeline stages

The pipeline runs in four stages, orchestrated by
[`wiki_pipeline/pipeline.py`](wiki_pipeline/pipeline.py):

| Stage | Module | What it does |
|-------|--------|--------------|
| 1. Fetch | [`wiki_pipeline/fetch/`](wiki_pipeline/fetch/) | Pull an article's full revision history from the MediaWiki API and split each revision into section-level edits. |
| 2. Linking | [`wiki_pipeline/linking/`](wiki_pipeline/linking/) | For each edit, find the previous version of that section by title + content similarity (scispaCy embeddings), so we know what to diff against. |
| 3. Prompts | [`wiki_pipeline/prompts/`](wiki_pipeline/prompts/) | Build a classification prompt per edit from the taxonomy, using either a "compare to previous" or "first-time section" template. |
| 4. Inference | [`wiki_pipeline/inference/`](wiki_pipeline/inference/) | Run the prompts through the OpenAI Batch API and parse the responses into labels + explanations. |

The taxonomy of change types lives in
[`wiki_pipeline/prompts/taxonomy/`](wiki_pipeline/prompts/taxonomy/).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# The linking stage needs the scispaCy model (not on PyPI):
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/en_core_sci_md-0.5.4.tar.gz
```

## Usage

Set your OpenAI key (required for the inference stage):

```bash
export OPENAI_API_KEY=your_api_key_here
```

Run the whole pipeline for an article:

```bash
./run_pipeline.sh "CRISPR" "results/CRISPR_classified.csv" "gpt-5-mini"
```

Equivalently, invoke the module directly:

```bash
python -m wiki_pipeline.pipeline \
    --article "CRISPR" \
    --output "results/CRISPR_classified.csv" \
    --model "gpt-5-mini"
```

The output CSV contains one row per section-level edit, with these columns:
`Article`, `Section`, `Revision Id`, `Timestamp`, `Author`,
`Previous Section Text` (the matched previous revision), `Edited Section Text`
(the current revision), and the model's `Labels` and `Explanations`.

> **Note on model names.** Models in the `gpt-5` family are reasoning models
> that reject a custom `temperature`; the client never sends one. Any
> chat-completions model available to your API key can be passed via
> `--model`.

## Using stages individually

Each stage is an importable function, so you can run them separately (e.g. run
the slow fetch + linking on one host and inference on another):

```python
import pandas as pd

from wiki_pipeline.fetch import fetch_revision_history
from wiki_pipeline.linking import add_flexible_revision_linking
from wiki_pipeline.prompts import (
    TAXONOMY_PATH, TAXONOMY_FIRST_TIME_PATH,
    create_taxonomy, generate_prompts_from_flexible_linking,
)
from wiki_pipeline.inference import GPT, parse_prompt

df, title = fetch_revision_history("CRISPR")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

linked = add_flexible_revision_linking(df, article_name=title)

taxonomy = create_taxonomy(TAXONOMY_PATH)
taxonomy_first = create_taxonomy(TAXONOMY_FIRST_TIME_PATH)
prompted, prompts = generate_prompts_from_flexible_linking(
    linked, title, taxonomy, taxonomy_first
)

responses = GPT(model_name="gpt-5-mini").predict(prompts)
parsed = [parse_prompt(r, taxonomy) for r in responses]
```

## Citation

```bibtex
@misc{ehrlich2026wikistar,
  title         = {WikiSTAR: A System for Shedding Light on the Hidden History
                   of Scientific Wikipedia Articles},
  author        = {Omer Ehrlich and Nitzan Barzilay and Rona Aviram and Tom Hope},
  year          = {2026},
  eprint        = {2607.12441},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.12441}
}
```
