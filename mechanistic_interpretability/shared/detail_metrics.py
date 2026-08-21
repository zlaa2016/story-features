from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
CAUSAL_CONNECTIVES = {
    "because", "cause", "caused", "causes", "consequently", "hence", "if", "so", "therefore", "thus", "when",
}
TEMPORAL_CONNECTIVES = {
    "after", "afterward", "afterwards", "before", "during", "earlier", "finally", "first", "later", "meanwhile",
    "next", "now", "once", "soon", "then", "until", "when", "while",
}
MENTAL_STATE_WORDS = {
    "believe", "believed", "feel", "felt", "forget", "forgot", "hope", "hoped", "imagine", "imagined", "know",
    "knew", "remember", "remembered", "think", "thought", "understand", "understood", "want", "wanted", "wish",
    "wished", "wonder", "wondered",
}
NEGATIONS = {"no", "not", "never", "neither", "nothing", "nowhere", "without", "won't", "wouldn't", "can't", "couldn't"}
PRONOUNS = {
    "he", "her", "hers", "herself", "him", "himself", "his", "i", "it", "itself", "me", "mine", "my", "myself",
    "our", "ours", "ourselves", "she", "their", "theirs", "them", "themselves", "they", "us", "we", "you", "your",
    "yours", "yourself", "yourselves",
}


def words(text: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_PATTERN.finditer(str(text or ""))]


def lcs_length(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def multiset_overlap(left: list[str], right: list[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def precision_recall_f1(overlap: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def lexical_preservation(reference_text: str, candidate_text: str) -> dict[str, float]:
    reference = words(reference_text)
    candidate = words(candidate_text)
    overlap = multiset_overlap(reference, candidate)
    token_precision, token_recall, token_f1 = precision_recall_f1(overlap, len(candidate), len(reference))
    lcs = lcs_length(reference, candidate)
    lcs_precision, lcs_recall, lcs_f1 = precision_recall_f1(lcs, len(candidate), len(reference))
    return {
        "lexical_token_precision": token_precision,
        "lexical_token_recall": token_recall,
        "lexical_token_f1": token_f1,
        "lexical_lcs_precision": lcs_precision,
        "lexical_lcs_recall": lcs_recall,
        "lexical_lcs_f1": lcs_f1,
    }


def heuristic_syllables(word: str) -> int:
    value = re.sub(r"[^a-z]", "", word.casefold())
    if not value:
        return 0
    groups = re.findall(r"[aeiouy]+", value)
    count = len(groups)
    if value.endswith("e") and not value.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def count_matches(tokens: Iterable[str], vocabulary: set[str]) -> int:
    return sum(token in vocabulary for token in tokens)


def surface_characteristics(text: str, syllable_lookup: dict[str, float] | None = None) -> dict[str, float]:
    token_list = words(text)
    n_words = len(token_list)
    sentence_count = max(1, len(re.findall(r"[.!?]+(?:\s|$)", str(text or ""))))
    unique = len(set(token_list))
    syllables = [
        float(syllable_lookup[token]) if syllable_lookup and token in syllable_lookup else float(heuristic_syllables(token))
        for token in token_list
    ]
    syllable_total = sum(syllables)
    words_per_sentence = n_words / sentence_count if sentence_count else 0.0
    syllables_per_word = syllable_total / n_words if n_words else 0.0
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word if n_words else math.nan
    fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59 if n_words else math.nan
    denominator = max(n_words, 1)
    return {
        "word_count": n_words,
        "sentence_count": sentence_count,
        "mean_word_length": sum(map(len, token_list)) / denominator,
        "type_token_ratio": unique / denominator,
        "syllables_per_word": syllables_per_word,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk_grade,
        "causal_connective_rate": count_matches(token_list, CAUSAL_CONNECTIVES) / denominator,
        "temporal_connective_rate": count_matches(token_list, TEMPORAL_CONNECTIVES) / denominator,
        "mental_state_rate": count_matches(token_list, MENTAL_STATE_WORDS) / denominator,
        "negation_rate": count_matches(token_list, NEGATIONS) / denominator,
        "pronoun_rate": count_matches(token_list, PRONOUNS) / denominator,
        "quoted_speech": float(bool(re.search(r'["“”]', str(text or "")))),
        "question": float("?" in str(text or "")),
        "exclamation": float("!" in str(text or "")),
    }


def aggregate_norm(tokens: list[str], values: dict[str, float], prefix: str) -> dict[str, float]:
    observed = [values[token] for token in tokens if token in values and math.isfinite(values[token])]
    return {
        f"{prefix}_mean": sum(observed) / len(observed) if observed else math.nan,
        f"{prefix}_coverage": len(observed) / max(len(tokens), 1),
    }
