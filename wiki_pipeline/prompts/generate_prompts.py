"""
Stage 3 (driver) — Turn linked revisions into per-row classification prompts.

Loads the taxonomy CSVs and, for each row produced by the linking stage,
builds either a "compare against previous" prompt or a "first-time section"
prompt depending on the flexible-linker match type.
"""
import pandas as pd

from wiki_pipeline.prompts.create_prompt import (
    create_prompt,
    create_prompt_for_first_time_section,
)


def create_taxonomy(taxonomy_csv_file_path):
    """Load a taxonomy CSV into a ``{label_number: {...}}`` mapping."""
    taxonomy_df = pd.read_csv(taxonomy_csv_file_path)
    taxonomy = {}
    for _, row in taxonomy_df.iterrows():
        key = int(row["#"])
        taxonomy[key] = {
            "category": row["Category Name"],
            "is_for_section_split": row["Section_Split"],
            "definition": row["Definition"],
            "example": row["Example"],
        }
    return taxonomy


def generate_prompts_from_flexible_linking(df, article_name, taxonomy, taxonomy_first_time):
    """Generate a classification prompt for every row using the linking columns.

    Args:
        df: DataFrame with the ``Flexible_*`` columns from stage 2.
        article_name: Article title (used in the prompt body).
        taxonomy: Taxonomy dict for revisions with a previous version.
        taxonomy_first_time: Taxonomy dict for first-time sections.

    Returns:
        ``(df, prompts)`` — the DataFrame with a ``Prompt`` column added, and
        the list of generated prompts (aligned with the DataFrame rows).
    """
    prompts = []

    for idx, row in df.iterrows():
        section_name = row["Section"]
        current_content = row["Changed Content"]
        match_type = row.get("Flexible_Match_Type", "")

        prev_content = row.get("Flexible_Previous_Content", "")
        prev_rev_id = row.get("Flexible_Previous_Revision_ID", "")

        if pd.isna(prev_content):
            prev_content = ""
        if pd.isna(prev_rev_id):
            prev_rev_id = ""

        # A section is "first time" when the linker found no comparable earlier
        # version (either an explicit first-time/none match, or no previous data).
        is_first_time = (
            match_type in ("first-time", "none")
            or (prev_content == "" and prev_rev_id == "")
        )

        if is_first_time:
            prompt = create_prompt_for_first_time_section(
                section_name, article_name, current_content, taxonomy_first_time
            )
        else:
            prompt = create_prompt(
                section_name, article_name, prev_content, current_content, taxonomy
            )

        prompts.append(prompt)

        if (idx + 1) % 100 == 0:
            print(f"Generated {idx + 1} prompts...")

    df["Prompt"] = prompts
    return df, prompts
