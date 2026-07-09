"""
Stage 4 (parsing) — Parse model responses into labels + explanations.

The model is asked to return a JSON object mapping label numbers (as strings)
to per-label explanations. ``parse_prompt`` validates that JSON against the
taxonomy and returns a normalized ``{"Labels": [...], "Explanation": {...}}``
structure, with a fallback for the older free-text response format.
"""
import json
import re

# Canonical label numbers → human-readable names.
label_mapping = {
    1: "New Information",
    2: "Clarification Added",
    3: "Researcher Names Added",
    4: "Technical Terms Added",
    5: "Information Removed",
    6: "Clarifications Removed",
    7: "Researcher Names Removed",
    8: "Technical Terms Removed",
    9: "Tense Change",
    10: "References Added",
    11: "No Significant Changes",
}


_OLD_LABELS_LINE_RE = re.compile(r"(?im)^\s*Labels?\s*:\s*(.+)$")
_OLD_EXPLANATION_LINE_RE = re.compile(
    r"(?is)Explanation\s*:\s*(.+?)(?:\n\s*Labels?\s*:|\Z)"
)


def _parse_old_prompt_text(response: str, taxonomy):
    """Fallback for the pre-JSON prompt format:

        Explanation: <free text>
        Labels: 1, 5, 6, 7, 9
    """
    if not isinstance(response, str) or not response.strip():
        return None

    labels_match = _OLD_LABELS_LINE_RE.search(response)
    if not labels_match:
        return None

    labels_text = labels_match.group(1).strip()
    labels_text = labels_text.rstrip(".;,")

    label_numbers = []
    # Split on comma/semicolon/newline only (NOT arbitrary whitespace), so that
    # verbose items like "1. New Scientific Information" stay a single item. Each
    # item is either a bare number, a "N. Category Name" pair, or (for a
    # comma-less line) a run of space-separated bare numbers like "3 4 6".
    for item in re.split(r"[,;\n]+", labels_text):
        item = item.strip().rstrip(".;,")
        if not item:
            continue
        if re.fullmatch(r"\d+(?:\s+\d+)*", item):
            tokens = item.split()
        else:
            leading = re.match(r"(\d+)", item)
            if not leading:
                return None
            tokens = [leading.group(1)]
        for tok in tokens:
            try:
                label_number = int(tok)
            except ValueError:
                return None
            if label_number not in taxonomy and label_number != 11:
                return None
            if label_number not in label_numbers:
                label_numbers.append(label_number)

    if not label_numbers:
        return None

    expl_match = _OLD_EXPLANATION_LINE_RE.search(response)
    explanation_text = expl_match.group(1).strip() if expl_match else ""
    explanation_map = {label: explanation_text for label in label_numbers}

    return {
        "Labels": sorted(label_numbers),
        "Explanation": explanation_map,
    }


def parse_prompt(response: str, taxonomy):
    """Parse a model response into ``{label: explanation}``.

    Label 11 ("No Significant Changes") is allowed even if not present in the
    taxonomy. If a label appears more than once, explanations are aggregated.

    Args:
        response: Raw model response (JSON string, optionally in a code fence).
        taxonomy: Dict of valid label numbers.

    Returns:
        ``{"Labels": [ints], "Explanation": {label: str}}`` or ``None`` if
        parsing fails or any label is invalid (except 11).
    """
    print("response: " + response)

    # Extract JSON from a markdown code block if present.
    json_match = re.search(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_content = json_match.group(1)
    else:
        json_content = response

    try:
        pairs = json.loads(json_content, object_pairs_hook=lambda pairs: pairs)
    except Exception:
        text_parsed = _parse_old_prompt_text(response, taxonomy)
        if text_parsed is not None:
            return text_parsed
        return None

    explanation_map = {}
    for key, explanation in pairs:
        try:
            label_number = int(key)
        except ValueError:
            # Try matching a category name instead of a number.
            found = False
            for num, tax_item in taxonomy.items():
                if key.strip().lower() == tax_item["category"].strip().lower():
                    label_number = num
                    found = True
                    break
            if not found:
                return None

        # Allow label 11 even if not in taxonomy.
        if label_number not in taxonomy and label_number != 11:
            return None

        explanation = explanation.strip()
        if label_number in explanation_map:
            explanation_map[label_number] += " " + explanation  # Aggregate
        else:
            explanation_map[label_number] = explanation

    if not explanation_map:
        return None

    return {
        "Labels": sorted(explanation_map.keys()),
        "Explanation": explanation_map,
    }
