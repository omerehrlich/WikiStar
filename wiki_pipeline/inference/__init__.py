"""Stage 4 — model inference and response parsing."""

from wiki_pipeline.inference.gpt import GPT
from wiki_pipeline.inference.parsers import label_mapping, parse_prompt

__all__ = ["GPT", "parse_prompt", "label_mapping"]
