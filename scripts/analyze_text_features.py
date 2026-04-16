#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PRONOUNS = {
    "i",
    "me",
    "my",
    "mine",
    "myself",
    "you",
    "your",
    "yours",
    "yourself",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "we",
    "us",
    "our",
    "ours",
    "ourselves",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
}


def _normalize_story_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (int, float)):
        try:
            return str(int(value)).zfill(6)
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"\d{1,6}", s):
        return s.zfill(6)
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists() and path.suffix.lower() == ".parquet":
        alt = path.with_suffix(".csv.gz")
        if alt.exists():
            path = alt
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            alt = path.with_suffix(".csv.gz")
            if alt.exists():
                return pd.read_csv(alt, low_memory=False)
            raise
    if path.suffix.lower() in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        df = pd.read_csv(path, low_memory=False)
        # Preserve leading zeros for story ids (review_samples use 6-digit ids).
        if "story_id" in df.columns:
            df["story_id"] = df["story_id"].map(_normalize_story_id)
        if "video_id" in df.columns:
            df["video_id"] = df["video_id"].astype(str)
        return df
    raise ValueError(f"Unsupported input: {path}")


def sent_split(text: str) -> List[str]:
    # Robust-ish sentence split for transcripts that may lack punctuation.
    sents = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sents if s and s.strip()]


def word_split(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())

_VOWELS = set("aeiouy")


def estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    # Remove silent trailing e
    if w.endswith("e"):
        w = w[:-1]
    groups = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            groups += 1
        prev_vowel = is_vowel
    return max(1, groups)


def readability_features(num_sentences: int, num_words: int, syllables: int) -> Dict[str, Any]:
    if num_sentences < 2 or num_words <= 0:
        return {
            "flesch_reading_ease_est": float("nan"),
            "flesch_kincaid_grade_est": float("nan"),
        }
    asl = num_words / num_sentences
    asw = syllables / num_words
    # Guard against pathological splits (e.g. 1 super-long "sentence")
    if asl > 80 or asw > 3.0:
        return {
            "flesch_reading_ease_est": float("nan"),
            "flesch_kincaid_grade_est": float("nan"),
            "avg_syllables_per_word": float(asw),
        }
    fre = 206.835 - 1.015 * asl - 84.6 * asw
    fkg = 0.39 * asl + 11.8 * asw - 15.59
    return {
        "flesch_reading_ease_est": float(fre),
        "flesch_kincaid_grade_est": float(fkg),
        "avg_syllables_per_word": float(asw),
    }


def lexical_diversity_features(words: List[str]) -> Dict[str, Any]:
    if not words:
        return {
            "ttr": float("nan"),
            "root_ttr": float("nan"),
            "log_ttr": float("nan"),
            "hapax_frac": float("nan"),
        }
    n = len(words)
    types = len(set(words))
    ttr = types / n
    root_ttr = types / math.sqrt(n)
    log_ttr = math.log(types + 1e-12) / math.log(n + 1e-12) if n > 1 and types > 0 else float("nan")
    counts = pd.Series(words).value_counts()
    hapax = int((counts == 1).sum())
    return {
        "ttr": float(ttr),
        "root_ttr": float(root_ttr),
        "log_ttr": float(log_ttr),
        "hapax_frac": float(hapax / n),
        "vocab_types": int(types),
    }


def sentiment_features(text: str) -> Dict[str, Any]:
    # Optional dependency; if not present, return NaNs.
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except Exception:
        return {
            "sentiment_compound": float("nan"),
            "sentiment_pos": float("nan"),
            "sentiment_neg": float("nan"),
            "sentiment_neu": float("nan"),
        }
    analyzer = SentimentIntensityAnalyzer()
    s = analyzer.polarity_scores(text)
    return {
        "sentiment_compound": float(s.get("compound", float("nan"))),
        "sentiment_pos": float(s.get("pos", float("nan"))),
        "sentiment_neg": float(s.get("neg", float("nan"))),
        "sentiment_neu": float(s.get("neu", float("nan"))),
    }


def load_aoa_lexicon(path: Path, word_col: str = "", aoa_col: str = "") -> Dict[str, float]:
    if not path.exists():
        return {}
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        sep = "\t" if suf in {".tsv"} else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
    cols = {c.lower().strip(): c for c in df.columns}
    if not word_col:
        for cand in ["word", "lemma", "item", "token"]:
            if cand in cols:
                word_col = cols[cand]
                break
    if not aoa_col:
        for cand in [
            "aoa",
            "aoa_mean",
            "mean_aoa",
            "rating.mean",  # e.g. Kuperman sheet
            "rating_mean",
            "aoa_kup_lem",  # AoA_51715_words.xlsx
            "aoa_kup",
            "aoa_kup_lemmas",
            "aoa_kup_lemma",
        ]:
            if cand in cols:
                aoa_col = cols[cand]
                break
    if not word_col or not aoa_col:
        return {}
    out: Dict[str, float] = {}
    for w, a in zip(df[word_col], df[aoa_col]):
        if pd.isna(w) or pd.isna(a):
            continue
        ws = str(w).strip().lower()
        try:
            out[ws] = float(a)
        except Exception:
            continue
    return out


def lemma_heuristic(word: str) -> str:
    w = word.lower()
    # possessives
    if w.endswith("'s"):
        w = w[:-2]
    if w.endswith("’s"):
        w = w[:-2]
    # simple plural handling
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    # simple verb inflections
    if len(w) > 5 and w.endswith("ing"):
        w2 = w[:-3]
        # running -> run (drop double consonant)
        if len(w2) >= 2 and w2[-1] == w2[-2]:
            w2 = w2[:-1]
        return w2
    if len(w) > 4 and w.endswith("ed"):
        w2 = w[:-2]
        if len(w2) >= 2 and w2[-1] == w2[-2]:
            w2 = w2[:-1]
        return w2
    return w


def aoa_features(words: List[str], aoa_map: Dict[str, float], lookup_mode: str = "wordform") -> Dict[str, Any]:
    if not aoa_map or not words:
        return {
            "aoa_mean": float("nan"),
            "aoa_std": float("nan"),
            "aoa_coverage": float("nan"),
        }
    vals: List[float] = []
    for w in words:
        key = w
        if lookup_mode == "lemma_heuristic":
            key = lemma_heuristic(w)
        a = aoa_map.get(key)
        if a is not None:
            vals.append(a)
    if not vals:
        return {
            "aoa_mean": float("nan"),
            "aoa_std": float("nan"),
            "aoa_coverage": 0.0,
        }
    s = pd.Series(vals, dtype="float")
    return {
        "aoa_mean": float(s.mean()),
        "aoa_std": float(s.std(ddof=0)),
        "aoa_coverage": float(len(vals) / len(words)),
    }


def frequency_features(words: List[str], corpus_counts: Dict[str, int], corpus_total: int) -> Dict[str, Any]:
    if not words or corpus_total <= 0:
        return {
            "mean_log10_corpus_freq": float("nan"),
            "rare_frac_corpus_le_5": float("nan"),
        }
    log_freqs: List[float] = []
    rare = 0
    for w in words:
        c = corpus_counts.get(w, 0)
        if c <= 5:
            rare += 1
        # add-one smoothing
        p = (c + 1) / (corpus_total + 1)
        log_freqs.append(math.log10(p))
    return {
        "mean_log10_corpus_freq": float(sum(log_freqs) / len(log_freqs)),
        "rare_frac_corpus_le_5": float(rare / len(words)),
    }


def basic_features(text: str) -> Dict[str, Any]:
    sents = sent_split(text)
    words = word_split(text)
    num_sents = len(sents)
    num_words = len(words)
    mean_sent_len = (num_words / num_sents) if num_sents else 0.0
    mean_word_len = (sum(len(w) for w in words) / num_words) if num_words else 0.0
    pronouns = sum(1 for w in words if w in PRONOUNS)
    return {
        "num_sentences": num_sents,
        "num_words": num_words,
        "mean_sentence_length_words": float(mean_sent_len),
        "mean_word_length_chars": float(mean_word_len),
        "num_pronouns": int(pronouns),
    }


def try_textstat(text: str) -> Dict[str, Any]:
    try:
        import textstat  # type: ignore
    except Exception:
        return {}
    try:
        return {"flesch_reading_ease": float(textstat.flesch_reading_ease(text))}
    except Exception:
        return {}


def _safe_json_loads(value: Any) -> Optional[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _event_text_and_counts(event: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(event, dict):
        return None
    sentences = event.get("sentences")
    if not isinstance(sentences, list):
        return None
    parts: List[str] = []
    for s in sentences:
        if isinstance(s, dict) and "text" in s:
            parts.append(str(s["text"]))
        elif isinstance(s, str):
            parts.append(s)
    text = " ".join(p.strip() for p in parts if p and str(p).strip()).strip()
    if not text:
        return None
    words = word_split(text)
    return {
        "text": text,
        "n_sentences": int(len(sentences)),
        "n_words": int(len(words)),
    }


def text_from_review_events(review_events_json: Any) -> Optional[str]:
    events = _safe_json_loads(review_events_json)
    if not isinstance(events, list) or not events:
        return None
    parts: List[str] = []
    for e in events:
        counts = _event_text_and_counts(e)
        if counts and counts.get("text"):
            parts.append(str(counts["text"]))
    joined = " ".join(parts).strip()
    return joined or None


def concept_features_from_events(review_events_json: Any) -> Dict[str, Any]:
    events = _safe_json_loads(review_events_json)
    if not isinstance(events, list) or not events:
        return {}

    type_counts: Dict[str, int] = {}
    seq: List[str] = []
    ev_word_counts: List[int] = []
    ev_sent_counts: List[int] = []

    for e in events:
        if not isinstance(e, dict):
            continue
        etype = e.get("type")
        etype_s = str(etype).strip() if etype is not None else ""
        if etype_s:
            type_counts[etype_s] = type_counts.get(etype_s, 0) + 1
            seq.append(etype_s)

        counts = _event_text_and_counts(e)
        if counts:
            ev_word_counts.append(int(counts["n_words"]))
            ev_sent_counts.append(int(counts["n_sentences"]))

    out: Dict[str, Any] = {}
    n_events = len(events)
    out["concept_n_events_parsed"] = int(n_events)
    if type_counts:
        out["concept_event_type_counts_json"] = json.dumps(type_counts, ensure_ascii=False)
        out["concept_event_type_seq"] = ">".join(seq)
        out["concept_event_type_unique"] = int(len(type_counts))
        out["concept_event_type_entropy"] = float(
            -sum((c / n_events) * math.log((c / n_events) + 1e-12) for c in type_counts.values())
        )

    if ev_word_counts:
        s = pd.Series(ev_word_counts, dtype="float")
        out["concept_mean_event_words"] = float(s.mean())
        out["concept_std_event_words"] = float(s.std(ddof=0))
        out["concept_median_event_words"] = float(s.median())
        out["concept_min_event_words"] = float(s.min())
        out["concept_max_event_words"] = float(s.max())
    if ev_sent_counts:
        s = pd.Series(ev_sent_counts, dtype="float")
        out["concept_mean_event_sentences"] = float(s.mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute low-level text features for stories and (optionally) events.")
    ap.add_argument("--master", type=Path, default=Path("data/processed/master_stories.parquet"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/text_features.csv"))
    ap.add_argument("--text-col", type=str, default="transcript_best")
    ap.add_argument("--id-col", type=str, default="video_id")
    ap.add_argument(
        "--text-from",
        type=str,
        choices=["transcript", "review_events"],
        default="transcript",
        help="Which text to analyze. 'review_events' concatenates sentences from review_samples (requires review_events_json).",
    )
    ap.add_argument(
        "--meta-cols",
        type=str,
        default="Title,Author,Categories,Num Reviews,Average Rating,Price,Grade Level,Publisher,Published Date,Language,Format,Print Length,ISBN-10,ISBN-13,Asin,Book URL,yt_view_count,yt_like_count,yt_duration,yt_uploader,yt_channel,yt_upload_date",
        help="Comma-separated metadata columns to carry through if present in master.",
    )
    ap.add_argument(
        "--concepts-from-events",
        action="store_true",
        help="Add narrative/event-level 'concept' features derived from review_events_json (if present).",
    )
    ap.add_argument(
        "--extra-nlp",
        action="store_true",
        help="Add non-LLM NLP features (readability, syllables, diversity, sentiment, corpus frequency).",
    )
    ap.add_argument(
        "--aoa-lexicon",
        type=str,
        default="",
        help="Optional path to an Age-of-Acquisition lexicon (CSV/TSV). If provided, adds AoA mean/std/coverage.",
    )
    ap.add_argument("--aoa-word-col", type=str, default="", help="Override AoA lexicon word column name.")
    ap.add_argument("--aoa-age-col", type=str, default="", help="Override AoA lexicon AoA column name.")
    ap.add_argument(
        "--aoa-lookup",
        type=str,
        choices=["wordform", "lemma_heuristic"],
        default="wordform",
        help="How to look up words in the AoA lexicon. Only affects AoA features.",
    )
    ap.add_argument(
        "--events-out",
        type=str,
        default="",
        help="Optional: write an event-level table (one row per event) to this path.",
    )
    ap.add_argument("--with-textstat", action="store_true", help="Add textstat metrics if available.")
    args = ap.parse_args()

    df = read_table(args.master)
    base_cols = [args.id_col, "story_id", "age_lo", "age_hi", "age_mid", args.text_col, "review_n_events"]
    if args.concepts_from_events:
        base_cols.append("review_events_json")
    if args.text_from == "review_events":
        base_cols.append("review_events_json")
    # De-duplicate in case the same column was added twice.
    base_cols = list(dict.fromkeys(base_cols))
    meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]
    cols = [c for c in base_cols + meta_cols if c in df.columns]
    work = df[cols].copy()

    work = work.rename(columns={args.id_col: "item_id", args.text_col: "text"})
    work["item_id"] = work["item_id"].astype(str)
    if args.text_from == "review_events":
        if "review_events_json" not in work.columns:
            raise ValueError("--text-from review_events requires review_events_json in the master table.")
        work["text"] = work["review_events_json"].apply(text_from_review_events)
        work = work[work["text"].notna()].reset_index(drop=True)
    else:
        work = work[work["text"].notna()].reset_index(drop=True)

    aoa_map: Dict[str, float] = {}
    if args.aoa_lexicon and str(args.aoa_lexicon).strip():
        aoa_path = Path(args.aoa_lexicon)
        if aoa_path.is_file():
            aoa_map = load_aoa_lexicon(aoa_path, args.aoa_word_col, args.aoa_age_col)

    corpus_counts: Dict[str, int] = {}
    corpus_total = 0
    if args.extra_nlp:
        for t in work["text"].astype(str).tolist():
            ws = word_split(t)
            corpus_total += len(ws)
            for w in ws:
                corpus_counts[w] = corpus_counts.get(w, 0) + 1

    carry_cols = ["age_lo", "age_hi", "age_mid", "review_n_events"] + meta_cols

    feats: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    for row in work.itertuples(index=False, name=None):
        row_map = dict(zip(work.columns, row))
        text = str(row_map["text"])
        f = basic_features(text)
        if args.with_textstat:
            f.update(try_textstat(text))

        if args.extra_nlp:
            words = word_split(text)
            syllables = sum(estimate_syllables(w) for w in words)
            f.update(lexical_diversity_features(words))
            f.update(readability_features(int(f.get("num_sentences", 0)), int(f.get("num_words", 0)), syllables))
            f["syllable_count_est"] = int(syllables)
            f.update(sentiment_features(text))
            f.update(frequency_features(words, corpus_counts, corpus_total))
            if aoa_map:
                f.update(aoa_features(words, aoa_map, lookup_mode=args.aoa_lookup))
        out: Dict[str, Any] = {"item_id": row_map.get("item_id"), "story_id": row_map.get("story_id")}
        for k in carry_cols:
            if k in row_map:
                out[k] = row_map.get(k)
        out.update(f)

        if args.concepts_from_events and "review_events_json" in row_map:
            cf = concept_features_from_events(row_map.get("review_events_json"))
            out.update(cf)

            if args.events_out and str(args.events_out).strip():
                events = _safe_json_loads(row_map.get("review_events_json"))
                if isinstance(events, list):
                    for e in events:
                        if not isinstance(e, dict):
                            continue
                        counts = _event_text_and_counts(e)
                        if not counts:
                            continue
                        event_rows.append(
                            {
                                "item_id": row_map.get("item_id"),
                                "story_id": row_map.get("story_id"),
                                "age_mid": row_map.get("age_mid"),
                                "event": e.get("event"),
                                "event_type": e.get("type"),
                                "boundary": e.get("boundary"),
                                "summary": e.get("summary"),
                                "event_text": counts["text"],
                                "event_n_sentences": counts["n_sentences"],
                                "event_n_words": counts["n_words"],
                            }
                        )

        feats.append(out)

    out_df = pd.DataFrame(feats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}  rows={len(out_df)}")

    if args.events_out and str(args.events_out).strip():
        ev_path = Path(args.events_out)
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(event_rows).to_csv(ev_path, index=False)
        print(f"Wrote: {ev_path}  rows={len(event_rows)}")


if __name__ == "__main__":
    main()
