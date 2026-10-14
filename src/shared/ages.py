from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgeCondition:
    name: str
    eligible: bool
    target_spec: str
    target_age: float | None
    skip_reason: str


def parse_age_bounds(value: Any) -> tuple[int, int] | None:
    numbers = [int(item) for item in re.findall(r"\d+", str(value or ""))]
    if not numbers:
        return None
    if len(numbers) == 1:
        return numbers[0], numbers[0]
    return min(numbers[0], numbers[1]), max(numbers[0], numbers[1])


def age_midpoint(value: Any) -> float | None:
    bounds = parse_age_bounds(value)
    return None if bounds is None else (bounds[0] + bounds[1]) / 2.0


def format_age(lower: int, upper: int) -> str:
    return str(lower) if lower == upper else f"{lower} - {upper}"


def relative_conditions(
    age_range: Any,
    *,
    minimum_age: int = 2,
    maximum_age: int = 18,
) -> list[AgeCondition]:
    bounds = parse_age_bounds(age_range)
    if bounds is None:
        return [
            AgeCondition(name, False, "", None, "missing_source_age")
            for name in ("younger", "actual", "older")
        ]
    lower, upper = bounds
    actual_lower = max(minimum_age, lower)
    actual_upper = min(maximum_age, upper)
    younger = lower - 3
    older = upper + 3
    return [
        AgeCondition(
            "younger",
            younger >= minimum_age,
            str(younger) if younger >= minimum_age else "",
            float(younger) if younger >= minimum_age else None,
            "" if younger >= minimum_age else "lower_bound_minus_3_below_minimum",
        ),
        AgeCondition(
            "actual",
            actual_lower <= actual_upper,
            format_age(actual_lower, actual_upper) if actual_lower <= actual_upper else "",
            (actual_lower + actual_upper) / 2.0 if actual_lower <= actual_upper else None,
            "" if actual_lower <= actual_upper else "source_range_outside_allowed_ages",
        ),
        AgeCondition(
            "older",
            older <= maximum_age,
            str(older) if older <= maximum_age else "",
            float(older) if older <= maximum_age else None,
            "" if older <= maximum_age else "upper_bound_plus_3_above_maximum",
        ),
    ]
