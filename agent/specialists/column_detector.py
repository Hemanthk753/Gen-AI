"""Specialist: Detects column-to-role mappings from uploaded experiment data.

Input:  column metadata + masked preview + user KPI + flow type
Output: ColumnDetectorOutput (via GPT structured output)
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from src.modules.agent.config import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from src.modules.agent.models.schemas import (
    ColumnDetectorOutput,
    ColumnMetadata,
    FlowType,
)
from src.modules.agent.prompts import COLUMN_DETECTOR_SYSTEM

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def detect_columns(
    columns: List[ColumnMetadata],
    total_rows: int,
    user_kpi: str,
    flow_type: FlowType,
    masked_preview: Optional[List[Dict[str, str]]] = None,
) -> ColumnDetectorOutput:
    """Detect column-to-role mappings for experiment data.

    Args:
        columns: List of ColumnMetadata with name, data_type, cardinality, null_pct
        total_rows: Total number of rows in the dataset
        user_kpi: User's selected KPI name (e.g., "Net Revenue")
        flow_type: FlowType.DESIGN or FlowType.MEASURE
        masked_preview: Optional masked sample rows for context

    Returns:
        ColumnDetectorOutput with granularity, date, kpi, group, features, blocking_factors
    """
    system_prompt = COLUMN_DETECTOR_SYSTEM.format(
        user_kpi=user_kpi,
        flow_type=flow_type.value,
        total_rows=total_rows,
    )

    # Build column summary
    col_summary = []
    for c in columns:
        col_summary.append({
            "name": c.name,
            "data_type": c.data_type,
            "cardinality": c.cardinality,
            "null_pct": round(c.null_pct, 3),
        })

    user_message = f"Columns:\n{json.dumps(col_summary, indent=2)}\n\nTotal rows: {total_rows}"

    if masked_preview:
        preview_str = json.dumps(masked_preview[:5], indent=2)
        user_message += f"\n\nSample data (masked):\n{preview_str}"

    response = await client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=ColumnDetectorOutput,
    )

    result = response.choices[0].message.parsed
    logger.info(
        f"Columns detected: granularity={result.granularity.column}, "
        f"date={result.date.column}, kpi={result.kpi.column}, "
        f"features={len(result.features)}, blocking={len(result.blocking_factors)}"
    )
    return result