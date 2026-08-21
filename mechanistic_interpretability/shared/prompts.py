from __future__ import annotations


def strict_summary_prompt(
    story_text: str,
    *,
    target_words: int,
    target_age: str,
    condition: str,
) -> str:
    if condition == "no_age":
        audience = "Do not adapt the summary to any stated audience age."
    else:
        audience = (
            f"Write for children in this target age setting: {target_age}. "
            "Use vocabulary and sentence structure appropriate for that age."
        )
    return f"""Summarize the story in exactly {target_words} words, allowing at most a 3-word deviation.

{audience}

Preserve the central plot, characters, causal relationships, and chronological order. Include only information explicitly present in the story. Do not mention the requested length or audience. Return only the summary text.

Story:
{story_text}
"""
