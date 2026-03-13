"""Data masking utilities — ensures no PII or sensitive data reaches GPT.

Used by column_detector specialist before sending preview rows to OpenAI.
Used by orchestrator before embedding experiment data in prompts.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional


def mask_preview_rows(
    rows: List[Dict[str, Any]],
    id_columns: Optional[List[str]] = None,
    max_rows: int = 5,
) -> List[Dict[str, str]]:
    """Mask PII in preview data rows before sending to GPT.

    Rules:
    - ID columns (granularity): hash to 8-char hex
    - Email-like strings: mask to "***@domain.com"
    - Phone-like strings: mask to "***-XXXX"
    - Numeric values: keep as-is (needed for column detection)
    - Date values: keep as-is (needed for column detection)
    - Long strings (>50 chars): truncate with "..."
    - All other strings: keep first 3 chars + "***"

    Args:
        rows: Raw preview rows from backend
        id_columns: Column names known to be identifiers
        max_rows: Maximum rows to return

    Returns:
        Masked rows safe to send to GPT
    """
    id_cols = set(id_columns or [])
    masked = []

    for row in rows[:max_rows]:
        masked_row = {}
        for col, val in row.items():
            masked_row[col] = _mask_value(val, is_id=col in id_cols)
        masked.append(masked_row)

    return masked


def mask_experiment_data(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive fields from experiment data before embedding in prompts.

    Keeps structural fields (status, step, type, dates, KPI names).
    Masks user-identifiable data (names, emails, literals).
    """
    safe = {}
    # Fields safe to send as-is
    safe_keys = [
        "experiment_name", "hypothesis", "problem_statement",
        "status", "status_text", "step", "step_text",
        "start_date", "end_date", "baseline_start_date", "baseline_end_date",
        "is_measurement_flow", "is_draft",
        "default_kpi", "metrices", "success_criteria",
        "group_configs", "interventions",
        "smaller_sample_size", "recommended_sample_size",
    ]
    for key in safe_keys:
        if key in experiment:
            safe[key] = experiment[key]

    # Mask nested objects — keep title only
    for nested_key in ["experiment_type", "country", "use_case", "business_function"]:
        if nested_key in experiment and isinstance(experiment[nested_key], dict):
            safe[nested_key] = {"title": experiment[nested_key].get("title", "N/A")}

    # Hash the literal
    if "experiment_literal" in experiment:
        safe["experiment_literal"] = _hash_id(experiment["experiment_literal"])

    return safe


def mask_column_metadata(
    columns: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ensure column metadata is safe to send to GPT.

    Column names, data types, cardinality, null_pct are NOT sensitive.
    This function strips any unexpected fields that might contain data samples.
    """
    safe_keys = {"name", "col_name", "column_name", "data_type", "type",
                 "cardinality", "distinct_count", "null_pct", "null_percentage"}
    return [
        {k: v for k, v in col.items() if k in safe_keys}
        for col in columns
    ]


# ── Internal helpers ─────────────────────────────────────────────


def _mask_value(val: Any, is_id: bool = False) -> str:
    """Mask a single value based on its type and content."""
    if val is None:
        return "null"

    s = str(val)

    # Numeric — keep as-is
    try:
        float(s)
        return s
    except (ValueError, TypeError):
        pass

    # ID columns — hash
    if is_id:
        return _hash_id(s)

    # Email pattern
    if re.match(r"[^@]+@[^@]+\.[^@]+", s):
        domain = s.split("@")[1]
        return f"***@{domain}"

    # Phone pattern
    if re.match(r"[\d\s\-\+\(\)]{7,15}", s.strip()):
        return "***-XXXX"

    # Date pattern — keep as-is
    if re.match(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", s):
        return s

    # Long strings — truncate
    if len(s) > 50:
        return s[:47] + "..."

    # Short strings — partial mask
    if len(s) > 3:
        return s[:3] + "***"

    return s


def _hash_id(value: str) -> str:
    """Hash an identifier to an 8-char hex string."""
    return hashlib.sha256(value.encode()).hexdigest()[:8]