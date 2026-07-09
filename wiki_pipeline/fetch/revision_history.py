"""
Stage 1 — Fetch a Wikipedia article's revision history from the MediaWiki API.

Walks every revision of an article (oldest first), splits each revision into
its sections, and records a row for every section whose content changed
relative to the previous revision. The result is a chronologically ordered
DataFrame of section-level edits, which the rest of the pipeline consumes.
"""
import re
import time

import pandas as pd
import requests

# MediaWiki API endpoint for the English Wikipedia.
API_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia rejects requests without a descriptive User-Agent (HTTP 403).
# https://meta.wikimedia.org/wiki/User-Agent_policy
HEADERS = {
    "User-Agent": (
        "WikiRevisionPipeline/0.1 "
        "(https://github.com/oehrlich/WikiColab; omer.ehrlich@mail.huji.ac.il)"
    )
}


def extract_sections_with_content(wikitext):
    """Split raw wikitext into a {section_title: {content, level}} mapping.

    The lead paragraph (before the first ``== heading ==``) is stored under the
    synthetic title ``(Top)``.
    """
    sections = {}
    current_section = "(Top)"
    current_level = 0
    content = []
    lines = wikitext.split("\n")
    for line in lines:
        # Match section headers (== section title ==)
        section_match = re.search(r"(==+)\s*(.*?)\s*(==+)", line)
        if section_match:
            # Save the previous section and its content if it exists
            if current_section is not None:
                sections[current_section] = {
                    "content": "\n".join(content).strip(),
                    "level": current_level,
                }
            # Start a new section
            current_section = section_match.group(2).split("/")[0].strip()
            current_level = len(section_match.group(1)) - 1
            content = []
        elif current_section is not None:
            content.append(line)

    # Save the last section if current_section is not None
    if current_section is not None:
        sections[current_section] = {
            "content": "\n".join(content).strip(),
            "level": current_level,
        }

    return sections


def fetch_revision_history(page_title):
    """Fetch the section-level edit history for a Wikipedia article.

    Args:
        page_title: Human-readable article title (e.g. "CRISPR").

    Returns:
        A tuple ``(section_history_df, normalized_title)``. On any failure
        (missing page, network error, no revisions) both elements are ``None``.
    """
    try:
        # First, check if the page exists.
        page_info_params = {
            "action": "query",
            "format": "json",
            "titles": page_title,
        }
        page_info_response = requests.get(API_URL, params=page_info_params, headers=HEADERS)
        page_info_response.raise_for_status()
        page_info_data = page_info_response.json()
        pages = page_info_data["query"]["pages"]
        page_id = next(iter(pages))
        if page_id == "-1":
            print(f"Page '{page_title}' does not exist. Skipping.")
            return None, None

        normalized_title = pages[page_id].get("title", page_title)
        print(f"Fetching revisions for page: {normalized_title}. Please wait...")

        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "titles": normalized_title,
            "rvprop": "ids|timestamp|user|content",
            "rvslots": "main",
            "rvlimit": "max",
            "rvdir": "newer",  # Start from the oldest revisions
        }

        section_history = []
        edit_counts = {}

        continuation = True
        previous_sections = {}
        revision_count = 0

        while continuation:
            try:
                response = requests.get(API_URL, params=params, headers=HEADERS)
                response.raise_for_status()
                data = response.json()

                pages = data["query"]["pages"]
                page_id = next(iter(pages))
                revisions = pages[page_id].get("revisions", [])

                if not revisions:
                    if revision_count == 0:
                        print(f"No revisions found for page: {normalized_title}")
                        return None, None
                    break

                for rev in revisions:
                    revision_count += 1
                    rev_id = rev["revid"]
                    timestamp = rev["timestamp"]
                    user = rev.get("user", "Anonymous/Bot")
                    slots = rev.get("slots", {})
                    main_slot = slots.get("main", {})
                    content = main_slot.get("*")  # Ensure we get the content safely

                    if content:
                        current_sections = extract_sections_with_content(content)

                        # Record every section whose content differs from the
                        # previous revision.
                        for section_title, section_info in current_sections.items():
                            section_content = section_info["content"]
                            section_level = section_info["level"]

                            if (
                                section_title not in previous_sections
                                or previous_sections[section_title]["content"] != section_content
                            ):
                                edit_info = {
                                    "Section": section_title,
                                    "Level": "*" * section_level,
                                    "Revision ID": rev_id,
                                    "Timestamp": timestamp,
                                    "User": user,
                                    "Changed Content": section_content,
                                }
                                section_history.append(edit_info)

                                if section_title in edit_counts:
                                    edit_counts[section_title] += 1
                                else:
                                    edit_counts[section_title] = 1

                        previous_sections = current_sections

                # Handle continuation (pagination).
                if "continue" in data:
                    params.update(data["continue"])
                else:
                    continuation = False

                # Small delay to avoid hitting rate limits.
                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                print(f"Network error while fetching revisions: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Error processing revisions: {e}")
                break

        section_history_df = pd.DataFrame(section_history)

        print(
            f"Found {revision_count} revisions and {len(edit_counts)} sections "
            f"for {normalized_title}"
        )

        return section_history_df, normalized_title

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None
