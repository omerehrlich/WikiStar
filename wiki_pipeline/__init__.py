"""Wikipedia revision-classification pipeline.

Four stages, run end-to-end by :mod:`wiki_pipeline.pipeline`:

1. ``fetch``     — pull an article's section-level revision history.
2. ``linking``   — find each section's previous revision by similarity.
3. ``prompts``   — build a classification prompt per revision.
4. ``inference`` — run the model and parse labels + explanations.
"""
