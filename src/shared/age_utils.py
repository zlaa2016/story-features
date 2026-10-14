from __future__ import annotations

import re
from typing import Optional


def normalize_age_text(value: object) -> str:
    bounds = parse_age_bounds(value)
    return format_age_range(*bounds) if bounds else ""


def parse_age_bounds(value: object) -> Optional[tuple[int, int]]:
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "[NA]", "NONE", "NAN"}:
        return None

    lowered = text.lower()
    nums = [int(x) for x in re.findall(r"\d+", lowered)]
    if not nums:
        return None

    if "baby" in lowered:
        upper = nums[0]
        if upper <= 1:
            return 1, 1
        return 1, upper

    if any(token in lowered for token in ["and up", "years and up", "year and up", "+"]):
        lower = nums[0]
        return lower, 18

    if len(nums) == 1:
        age = nums[0]
        return age, age

    lower, upper = nums[0], nums[1]
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def parse_age_midpoint(value: object) -> Optional[float]:
    bounds = parse_age_bounds(value)
    if not bounds:
        return None
    lower, upper = bounds
    return float(lower) if lower == upper else (lower + upper) / 2.0


def format_age_range(lower: int, upper: int) -> str:
    return f"{lower}" if lower == upper else f"{lower} - {upper}"
