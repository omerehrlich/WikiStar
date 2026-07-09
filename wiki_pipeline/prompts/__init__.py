"""Stage 3 — build classification prompts."""

import os

from wiki_pipeline.prompts.create_prompt import (
    create_prompt,
    create_prompt_for_first_time_section,
)
from wiki_pipeline.prompts.generate_prompts import (
    create_taxonomy,
    generate_prompts_from_flexible_linking,
)

# Directory holding the bundled taxonomy CSVs.
TAXONOMY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy")
TAXONOMY_PATH = os.path.join(TAXONOMY_DIR, "taxonomy.csv")
TAXONOMY_FIRST_TIME_PATH = os.path.join(TAXONOMY_DIR, "taxonomy_first_time.csv")

__all__ = [
    "create_prompt",
    "create_prompt_for_first_time_section",
    "create_taxonomy",
    "generate_prompts_from_flexible_linking",
    "TAXONOMY_DIR",
    "TAXONOMY_PATH",
    "TAXONOMY_FIRST_TIME_PATH",
]
