"""
Stage 2 — Flexible revision linking.

For each section-level edit, find the most likely *previous* version of that
section to compare against. Rather than assuming a section keeps a stable
title, this matches on both title similarity (Levenshtein) and content
similarity (scispaCy embeddings), looking back over a configurable window of
earlier edits. The chosen match supplies the "previous revision" the prompt
stage diffs against.
"""
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import spacy
from nltk.metrics.distance import edit_distance
from spacy.tokens import Doc

from wiki_pipeline.constants import CHANGED_CONTENT

# spaCy model used for content-similarity matching. Install with:
#   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/\
#   v0.5.4/en_core_sci_md-0.5.4.tar.gz
SPACY_MODEL = "en_core_sci_md"


class FlexibleRevisionLinker:
    """Find previous revisions of a section based on similarity matching."""

    def __init__(
        self,
        title_threshold: float = 0.8,
        content_threshold: float = 0.99,
        lookback_window: int = 1000,
    ):
        """
        Args:
            title_threshold: Levenshtein similarity threshold for title matching.
            content_threshold: Semantic similarity threshold for content matching.
            lookback_window: Number of previous revisions to search.
        """
        self.title_threshold = title_threshold
        self.content_threshold = content_threshold
        self.lookback_window = lookback_window

        # Load spaCy model for semantic similarity.
        self.nlp = spacy.load(SPACY_MODEL)
        self.nlp.max_length = 10000

        # Embedding cache for efficiency (key: "revision_id_section_name").
        self.embedding_cache: Dict[str, Doc] = {}
        self.processed_items: List[Dict] = []

    def levenshtein_similarity(self, title_a: str, title_b: str) -> float:
        """Levenshtein similarity between two titles, normalized to [0, 1]."""
        title_a, title_b = str(title_a), str(title_b)
        distance = edit_distance(title_a, title_b)
        max_length = max(len(title_a), len(title_b))
        return 1 - (distance / max_length) if max_length > 0 else 0

    def get_content_embedding(
        self, revision_id: int, content: str, section_name: str = ""
    ) -> Doc:
        """Get or compute a content embedding, with caching."""
        if pd.isna(content) or content is None:
            content = ""
        else:
            content = str(content)

        if len(content) > self.nlp.max_length:
            content = content[: self.nlp.max_length]

        # Cache key of revision ID + section name: the same revision may touch
        # several sections, each needing its own embedding.
        cache_key = f"{revision_id}_{section_name}"

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        doc = self.nlp(content)
        self.embedding_cache[cache_key] = doc
        return doc

    def find_previous_revision(self, current_item: Dict) -> Tuple[Optional[Dict], str]:
        """Find the best previous revision for the current item.

        Returns:
            ``(previous_item_dict, match_type)`` where match_type is one of
            'content', 'title', 'both', 'none', or 'first-time'.
        """
        current_section = current_item["Section"]
        current_content = current_item[CHANGED_CONTENT]
        current_rev_id = current_item["Revision ID"]

        # First revision case.
        if len(self.processed_items) == 0:
            return None, "first-time"

        # Special case: (Top) section should only match with other (Top) sections.
        if current_section == "(Top)":
            return self._find_top_section_match(current_item)

        current_doc = self.get_content_embedding(
            current_rev_id, current_content, current_section
        )

        # Search through the lookback window, most recent to oldest.
        lookback_start = max(0, len(self.processed_items) - self.lookback_window)
        search_items = self.processed_items[lookback_start:]

        for i, prev_item in enumerate(reversed(search_items)):
            # Skip if same revision (safety check).
            if prev_item["Revision ID"] == current_rev_id:
                continue

            title_match = False
            content_match = False

            title_sim = self.levenshtein_similarity(current_section, prev_item["Section"])
            if title_sim >= self.title_threshold:
                title_match = True

            prev_doc = self.get_content_embedding(
                prev_item["Revision ID"], prev_item[CHANGED_CONTENT], prev_item["Section"]
            )
            content_sim = current_doc.similarity(prev_doc)
            if content_sim >= self.content_threshold:
                content_match = True

            # Return immediately on the first match found.
            if title_match and content_match:
                return prev_item, "both"
            elif content_match:
                return prev_item, "content"
            elif title_match:
                return prev_item, "title"

        # No match found in the entire lookback window.
        return None, "none"

    def _find_top_section_match(self, current_item: Dict) -> Tuple[Optional[Dict], str]:
        """Match logic for (Top) sections — only match with other (Top) sections."""
        current_content = current_item[CHANGED_CONTENT]
        current_rev_id = current_item["Revision ID"]

        current_doc = self.get_content_embedding(current_rev_id, current_content, "(Top)")

        lookback_start = max(0, len(self.processed_items) - self.lookback_window)
        search_items = self.processed_items[lookback_start:]

        # Prefer a content match against another (Top) section.
        for prev_item in reversed(search_items):
            if prev_item["Revision ID"] == current_rev_id:
                continue
            if prev_item["Section"] != "(Top)":
                continue

            prev_doc = self.get_content_embedding(
                prev_item["Revision ID"], prev_item[CHANGED_CONTENT], prev_item["Section"]
            )
            content_sim = current_doc.similarity(prev_doc)

            if content_sim >= self.content_threshold:
                # (Top) sections always share a title, so a content match is 'both'.
                return prev_item, "both"

        # Otherwise fall back to any earlier (Top) section (title match only).
        for prev_item in reversed(search_items):
            if prev_item["Revision ID"] == current_rev_id:
                continue
            if prev_item["Section"] == "(Top)":
                return prev_item, "title"

        return None, "none"

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add flexible revision linking columns to a chronologically sorted df."""
        df = df.copy()

        # Initialize result columns as object dtype so later assignments can mix
        # strings, ints, and NaN without pandas raising on a StringArray.
        df["Flexible_Previous_Content"] = pd.Series([""] * len(df), dtype=object, index=df.index)
        df["Flexible_Previous_Revision_ID"] = pd.Series([""] * len(df), dtype=object, index=df.index)
        df["Flexible_Previous_Section"] = pd.Series([""] * len(df), dtype=object, index=df.index)
        df["Flexible_Match_Type"] = pd.Series([""] * len(df), dtype=object, index=df.index)
        df["Flexible_Match_Distance"] = 0

        # Reset internal state.
        self.processed_items = []
        self.embedding_cache = {}

        print(f"Processing {len(df)} revisions with flexible linking...")

        for idx, row in df.iterrows():
            current_item = {
                "Section": row["Section"],
                "Revision ID": row["Revision ID"],
                CHANGED_CONTENT: row[CHANGED_CONTENT],
                "Timestamp": row["Timestamp"],
                "index": idx,
            }

            prev_item, match_type = self.find_previous_revision(current_item)

            if prev_item is not None:
                df.at[idx, "Flexible_Previous_Content"] = prev_item[CHANGED_CONTENT]
                df.at[idx, "Flexible_Previous_Revision_ID"] = prev_item["Revision ID"]
                df.at[idx, "Flexible_Previous_Section"] = prev_item["Section"]
                df.at[idx, "Flexible_Match_Type"] = match_type
                df.at[idx, "Flexible_Match_Distance"] = idx - prev_item["index"]
            else:
                df.at[idx, "Flexible_Match_Type"] = match_type
                df.at[idx, "Flexible_Match_Distance"] = 0

            self.processed_items.append(current_item)

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} revisions...")

        return df

    def get_stats(self) -> Dict:
        """Statistics about the linking process."""
        return {
            "cache_size": len(self.embedding_cache),
            "processed_items": len(self.processed_items),
            "title_threshold": self.title_threshold,
            "content_threshold": self.content_threshold,
            "lookback_window": self.lookback_window,
        }


def add_flexible_revision_linking(
    df: pd.DataFrame,
    article_name: str,
    title_threshold: float = 0.8,
    content_threshold: float = 0.99,
    lookback_window: int = 1000,
) -> pd.DataFrame:
    """Add flexible revision linking columns to a section-history DataFrame.

    Args:
        df: DataFrame from stage 1 (must be sorted chronologically).
        article_name: Article title (for logging).
        title_threshold: Levenshtein similarity threshold for title matching.
        content_threshold: Semantic similarity threshold for content matching.
        lookback_window: Number of previous revisions to search.

    Returns:
        The DataFrame with ``Flexible_*`` columns added.
    """
    print("=" * 60)
    print("FLEXIBLE REVISION LINKING")
    print("=" * 60)
    print("Configuration:")
    print(f"  Title threshold: {title_threshold}")
    print(f"  Content threshold: {content_threshold}")
    print(f"  Lookback window: {lookback_window}")
    print(f"  Input revisions: {len(df)}")
    print(f"  Article title: '{article_name}'")

    linker = FlexibleRevisionLinker(title_threshold, content_threshold, lookback_window)
    enhanced_df = linker.process_dataframe(df)

    # Print statistics.
    stats = linker.get_stats()
    match_type_counts = enhanced_df["Flexible_Match_Type"].value_counts()

    print("\nResults:")
    print(f"  Total revisions: {len(enhanced_df)}")
    for match_type, count in match_type_counts.items():
        percentage = (count / len(enhanced_df)) * 100
        print(f"  {match_type.capitalize()} matches: {count} ({percentage:.1f}%)")

    successful_matches = enhanced_df[enhanced_df["Flexible_Match_Distance"] > 0]
    if len(successful_matches) > 0:
        avg_distance = successful_matches["Flexible_Match_Distance"].mean()
        max_distance = successful_matches["Flexible_Match_Distance"].max()
        print(f"  Average match distance: {avg_distance:.1f} revisions")
        print(f"  Maximum match distance: {max_distance} revisions")

    print(f"  Embeddings cached: {stats['cache_size']}")

    print("=" * 60)
    print("FLEXIBLE REVISION LINKING COMPLETE")
    print("=" * 60)

    return enhanced_df
