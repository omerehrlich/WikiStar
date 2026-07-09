"""
Stage 3 (core) — Build the classification prompt shown to the model.

Two variants:
  * ``create_prompt`` — compares a section's current revision against a
    previous one (used when the linker found a match).
  * ``create_prompt_for_first_time_section`` — describes a section the first
    time it appears, with no previous revision to diff against.

Both render the taxonomy of change types and ask the model to return a JSON
object mapping applicable label numbers to per-label explanations.
"""


def create_prompt(section_name, article_name, previous_revision, current_revision, taxonomy):
    """Prompt comparing a current revision of a section against its previous one."""
    labels_text = "\n".join(
        [
            f"{num}. {taxonomy[num]['category']}: "
            f"{taxonomy[num]['definition'].replace('{PAGE_NAME}', article_name)}\n"
            f"\tExample: {taxonomy[num]['example']}"
            for num in sorted(taxonomy.keys())
        ]
    )
    prompt = f"""We are going to show you two revisions of the section "{section_name}" from the Wikipedia article "{article_name}".
Your task is to compare the current revision to the previous one and determine whether it has scientific importance
for the scientific representation of {article_name}.
You should select from a predefined list which types of details make the current revision scientifically interesting —
 you may choose one or more tags as needed.

Please indicate all tags relevant to the current revision.

If no meaningful scientific modifications are detected between the current and previous revision, choose Label 11.
This includes changes that do not affect scientific content, such as template/tag modifications ({{clarify}}, {{citation needed}}, etc.),
reference formatting changes without changing the source, addition/removal of non-scientific content or vandalism, URL
 additions outside of <ref> tags, wikilink formatting changes, changes to existing reference metadata, minor rephrasings,
 wording adjustments, or when the two revisions are effectively the same. This label should only be used alone, not together with other labels.

Previous revision: {previous_revision}

Current revision: {current_revision}

{labels_text}

Instructions:
- Only include labels that have **clear, specific evidence** of a change.
- Do **not** include labels that describe what did *not* change (e.g., "No clarification added").
- If a label applies, explain exactly **what changed** using **explicit text from the revision** — e.g., quote the term, phrase, or name added/removed.
- Be detailed and concrete. Avoid vague explanations like "No change detected" or "Nothing was added."
- If and only if **no substantive changes** are found, return:
  {{
      "11": "Explanation of why there is no significant change, referencing both versions."
  }}

Response format:
First, think step-by-step for each category to ensure you choose the best label.
Then output only the final JSON in the exact format above.

Use **only** the keys (as strings) for the relevant label numbers, and the values as detailed explanations.

Correct example when a technical term is added:
{{
    "4": "The term 'CRISPR-Cas9' was introduced in the current revision, representing a new technical term."
}}

Correct example when no significant change occurred:
{{
    "11": "The current and previous revisions contain minor rephrasings without any new information or removed content."
}}

"""
    return prompt


def create_prompt_for_first_time_section(section_name, article_name, current_revision, taxonomy):
    """Prompt for the first time a section appears (no previous revision)."""
    labels_text = "\n".join(
        [
            f"{num}. {taxonomy[num]['category']}: "
            f"{taxonomy[num]['definition'].replace('{PAGE_NAME}', article_name)}\n"
            f"\tExample: {taxonomy[num]['example']}"
            for num in sorted(taxonomy.keys())
        ]
    )

    prompt = f"""We are going to show you a current revision of the section "{section_name}" from the Wikipedia article "{article_name}".
Your task is to analyze this revision and determine whether it contains scientifically meaningful information relevant to the scientific representation of "{article_name}".
 A change is considered scientifically relevant if it modifies how scientific information is presented, accessed, or understood in the article.
  Below you will find a detailed taxonomy of changes that are considered scientifically important, including the addition of wiki links and technical terms.
You should select from a predefined list which types of details make the current revision scientifically interesting — you may choose one or more tags as needed.

Please indicate all tags relevant to the current revision.

If no meaningful scientific modifications are detected in the current revision, choose Label 11.
This includes additions that do not affect scientific content, such as template/tag additions ({{clarify}}, {{citation needed}}, etc.),
addition of non-scientific content or vandalism, URL additions outside of <ref> tags.
This label should only be used alone, not together with other labels.

Current revision: {current_revision}

{labels_text}


Instructions:
- Only include labels that have **clear, specific evidence** of a change.
- Do **not** include labels that describe what did *not* change (e.g., "No clarification added").
- If a label applies, explain exactly **what changed** using **explicit text from the revision** — e.g., quote the term, phrase, or name added/removed.
- Be detailed and concrete. Avoid vague explanations like "No change detected" or "Nothing was added."
- If and only if **no substantive changes** are found, return:
  {{
      "11": "Explanation of why there is no significant change, referencing both versions."
  }}

Response format:
First, think step-by-step for each category to ensure you choose the best label.
Then output only the final JSON in the exact format above.

Use **only** the keys (as strings) for the relevant label numbers, and the values as detailed explanations.

Correct example when a technical term is added:
{{
    "4": "The term 'CRISPR-Cas9' was introduced in the current revision, representing a new technical term."
}}

Correct example when no significant change occurred:
{{
    "11": "The current and previous revisions contain minor rephrasings without any new information or removed content."
}}

   """
    return prompt
