"""
eval/metrics.py — Evaluation Metric Definitions
================================================
All metric functions are pure (no I/O, no API calls).

Metrics implemented:
  answer_relevance    LLM-as-judge 0–4 score from Gemini
  filter_accuracy     % of expected filters correctly extracted by /ask
  keyword_hit_rate    % of expected_answer_keywords found in the answer
  has_results         boolean — did the endpoint return any listings?
  retrieved_count     integer — how many listings were returned
  latency_ms          float — wall-clock time of the API call

Scoring rubric (answer_relevance):
  4 — Fully answers the query; all key details present; no hallucination
  3 — Mostly answers; minor omissions or vague details
  2 — Partially answers; some relevant info but significant gaps
  1 — Barely relevant; mostly off-topic
  0 — Completely irrelevant or refused to answer
"""
from __future__ import annotations

import re
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# Keyword hit rate
# ═══════════════════════════════════════════════════════════════════════════════

def keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float:
    """
    Fraction of expected_keywords that appear (case-insensitive) in the answer.

    Returns 1.0 if expected_keywords is empty (no constraint → auto-pass).
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Filter accuracy  (/ask only)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_room_type(val: str | None) -> str | None:
    """Normalise room_type strings for fuzzy matching."""
    if val is None:
        return None
    v = val.lower().strip()
    if "entire" in v or "whole" in v:
        return "entire home/apt"
    if "private" in v:
        return "private room"
    if "shared" in v:
        return "shared room"
    if "hotel" in v:
        return "hotel room"
    return v


def filter_accuracy(tool_calls: list[dict], expected_filters: dict) -> float:
    """
    Measure how accurately /ask extracted the expected filters.

    Inspects tool_calls (list of {"name": "find_rentals", "args": {...}})
    and checks that each expected filter appears in at least one call.

    Returns:
        1.0  if all expected filters were correctly extracted
        0.0  if no tool calls were made (or expected_filters is empty → 1.0)
        0-1  partial credit proportional to filters matched
    """
    if not expected_filters:
        return 1.0      # no filter expectations → auto-pass

    if not tool_calls:
        return 0.0      # filters expected but agent made no tool calls

    # Collect all args from every find_rentals call
    all_args: dict[str, Any] = {}
    for call in tool_calls:
        if call.get("name") == "find_rentals":
            all_args.update(call.get("args", {}))

    matched = 0
    for key, expected_val in expected_filters.items():
        extracted_val = all_args.get(key)
        if extracted_val is None:
            continue

        # Numeric tolerance: within 20 %
        if isinstance(expected_val, (int, float)):
            try:
                if abs(float(extracted_val) - float(expected_val)) <= abs(float(expected_val)) * 0.2:
                    matched += 1
            except (TypeError, ValueError):
                pass

        # Boolean exact match
        elif isinstance(expected_val, bool):
            if bool(extracted_val) == expected_val:
                matched += 1

        # String: room_type normalised, neighbourhood substring
        elif isinstance(expected_val, str):
            if key == "room_type":
                if _normalise_room_type(str(extracted_val)) == _normalise_room_type(expected_val):
                    matched += 1
            elif key == "neighbourhood":
                if expected_val.lower() in str(extracted_val).lower():
                    matched += 1
            else:
                if str(extracted_val).lower() == expected_val.lower():
                    matched += 1

    return round(matched / len(expected_filters), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-as-judge answer relevance (called from evaluate.py with a live model)
# ═══════════════════════════════════════════════════════════════════════════════

JUDGE_PROMPT = """\
You are an expert evaluator for an Airbnb listing search assistant in Austin, Texas.

Rate the ANSWER on a scale of 0 to 4 using this rubric:
  4 — Fully answers the query; all key details present; no hallucination
  3 — Mostly answers; minor omissions or vague details
  2 — Partially answers; some relevant info but significant gaps
  1 — Barely relevant; mostly off-topic or unhelpful
  0 — Completely irrelevant, refused to answer, or obvious hallucination

USER QUERY:
{query}

ANSWER:
{answer}

Respond with ONLY a single integer (0, 1, 2, 3, or 4). No explanation.
"""


def build_judge_prompt(query: str, answer: str) -> str:
    return JUDGE_PROMPT.format(query=query, answer=answer)


def parse_judge_score(raw: str) -> int | None:
    """Extract the integer score from the judge LLM's response."""
    m = re.search(r"\b([0-4])\b", raw.strip())
    return int(m.group(1)) if m else None


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregate summary helpers
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate(results: list[dict]) -> dict:
    """
    Compute aggregate statistics over all eval results.

    Args:
        results: list of per-query result dicts (from evaluate.py)

    Returns:
        dict with avg/min/max for each metric, plus per-category breakdown
    """
    def safe_avg(vals):
        vs = [v for v in vals if v is not None]
        return round(sum(vs) / len(vs), 4) if vs else None

    def safe_pct(vals):
        vs = [v for v in vals if v is not None]
        return round(sum(1 for v in vs if v) / len(vs), 4) if vs else None

    rag_rows = [r for r in results if r.get("endpoint") == "rag"]
    ask_rows = [r for r in results if r.get("endpoint") == "ask"]

    def stats_for(rows, tag):
        if not rows:
            return {}
        return {
            f"{tag}_count":               len(rows),
            f"{tag}_avg_relevance":       safe_avg([r.get("answer_relevance") for r in rows]),
            f"{tag}_avg_keyword_hit_rate":safe_avg([r.get("keyword_hit_rate")  for r in rows]),
            f"{tag}_has_results_pct":     safe_pct([r.get("has_results")       for r in rows]),
            f"{tag}_avg_latency_ms":      safe_avg([r.get("latency_ms")        for r in rows]),
            f"{tag}_avg_retrieved":       safe_avg([r.get("retrieved_count")   for r in rows]),
        }

    summary = {
        "total_queries": len(results),
        **stats_for(rag_rows, "rag"),
        **stats_for(ask_rows, "ask"),
        "ask_avg_filter_accuracy": safe_avg([r.get("filter_accuracy") for r in ask_rows]),
    }

    # Per-category breakdown
    categories = sorted({r.get("category", "unknown") for r in results})
    category_stats = {}
    for cat in categories:
        cat_rows = [r for r in results if r.get("category") == cat]
        category_stats[cat] = {
            "count":            len(cat_rows),
            "avg_relevance":    safe_avg([r.get("answer_relevance") for r in cat_rows]),
            "avg_latency_ms":   safe_avg([r.get("latency_ms")       for r in cat_rows]),
            "has_results_pct":  safe_pct([r.get("has_results")       for r in cat_rows]),
        }
    summary["by_category"] = category_stats

    return summary
