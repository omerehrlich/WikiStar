"""
End-to-end pipeline orchestrator.

    1) Fetch Wikipedia revisions for an article          (stage: fetch)
    2) Flexible section linking                          (stage: linking)
    3) Build classification prompts                      (stage: prompts)
    4) Run model predictions, parse them, save the CSV   (stage: inference)

Only the final CSV is written to disk — intermediate stages stay in-memory.
"""
import argparse
import os
import sys

import pandas as pd

from wiki_pipeline.fetch import fetch_revision_history
from wiki_pipeline.linking import add_flexible_revision_linking
from wiki_pipeline.prompts import (
    TAXONOMY_FIRST_TIME_PATH,
    TAXONOMY_PATH,
    create_taxonomy,
    generate_prompts_from_flexible_linking,
)
from wiki_pipeline.inference import GPT, parse_prompt


def run_pipeline(article_name: str, output_path: str, model_name: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: export OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)

    print("=" * 60)
    print(f"Step 1/4: Fetching Wikipedia revisions for '{article_name}'")
    print("=" * 60)
    section_history_df, normalized_title = fetch_revision_history(article_name)
    if section_history_df is None or section_history_df.empty:
        print(f"No revisions retrieved for '{article_name}'. Aborting.")
        sys.exit(1)
    print(f"Retrieved {len(section_history_df)} revisions for '{normalized_title}'.")

    section_history_df["Timestamp"] = pd.to_datetime(section_history_df["Timestamp"])
    section_history_df = section_history_df.sort_values("Timestamp").reset_index(drop=True)

    print("=" * 60)
    print("Step 2/4: Flexible section linking")
    print("=" * 60)
    linked_df = add_flexible_revision_linking(
        section_history_df,
        article_name=normalized_title,
        title_threshold=0.8,
        content_threshold=0.99,
        lookback_window=1000,
    )

    print("=" * 60)
    print("Step 3/4: Building prompts")
    print("=" * 60)
    taxonomy = create_taxonomy(TAXONOMY_PATH)
    taxonomy_first_time = create_taxonomy(TAXONOMY_FIRST_TIME_PATH)
    prompted_df, prompts = generate_prompts_from_flexible_linking(
        linked_df, normalized_title, taxonomy, taxonomy_first_time
    )
    print(f"Generated {len(prompts)} prompts.")

    print("=" * 60)
    print(f"Step 4/4: Running model predictions with model '{model_name}'")
    print("=" * 60)
    model = GPT(model_name=model_name)
    raw_responses = model.predict(prompts)

    labels_col = [None] * len(prompted_df)
    explanation_col = [None] * len(prompted_df)
    parsed_ok = 0
    for i, response in enumerate(raw_responses):
        parsed = parse_prompt(response, taxonomy)
        if parsed is None:
            print(f"[warn] Failed to parse response at row {i}")
            continue
        labels_col[i] = parsed["Labels"]
        explanation_col[i] = parsed["Explanation"]
        parsed_ok += 1
    print(f"Parsed {parsed_ok}/{len(prompts)} responses successfully.")

    prompted_df["Labels"] = labels_col
    prompted_df["Explanation"] = explanation_col

    # Insert Labels/Explanation right after the Changed Content column.
    cols = list(prompted_df.columns)
    cols.remove("Labels")
    cols.remove("Explanation")
    insert_after = "Changed Content" if "Changed Content" in cols else cols[-1]
    insert_at = cols.index(insert_after) + 1
    cols = cols[:insert_at] + ["Labels", "Explanation"] + cols[insert_at:]
    prompted_df = prompted_df[cols]

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    prompted_df.to_csv(output_path, index=False)
    print(f"\nFinal results saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Wikipedia-revision classification pipeline."
    )
    parser.add_argument(
        "--article",
        required=True,
        help="Name of the Wikipedia article to fetch revisions for (e.g. 'CRISPR').",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output CSV with model predictions.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Model name (default: gpt-5-mini).",
    )
    args = parser.parse_args()

    run_pipeline(args.article, args.output, args.model)


if __name__ == "__main__":
    main()
