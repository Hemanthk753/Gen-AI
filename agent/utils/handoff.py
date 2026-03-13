"""Specialist handoff — routes tasks to the correct specialist.

Extracted from orchestrator to keep routing logic separate from flow logic.
The orchestrator calls handoff functions instead of specialists directly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from src.modules.agent.utils.privacy import mask_preview_rows
from src.modules.agent.config import MAX_SPECIALIST_RETRIES
from src.modules.agent.models.schemas import (
    ColumnDetectorOutput,
    ColumnMetadata,
    DateValidatorOutput,
    FlowType,
    HypothesisParserOutput,
    UpdateValidatorOutput,
)
from src.modules.agent.specialists.column_detector import detect_columns
from src.modules.agent.specialists.date_validator import validate_dates
from src.modules.agent.specialists.hypothesis_parser import parse_hypothesis
from src.modules.agent.specialists.update_validator import validate_update
from src.modules.agent.utils.privacy import mask_preview_rows

logger = logging.getLogger(__name__)


async def handoff_hypothesis(
    hypothesis_text: str,
    available_kpis: List[str],
    available_experiment_types: List[str],
) -> HypothesisParserOutput:
    """Route to hypothesis parser with retry logic.

    Args:
        hypothesis_text: User's raw hypothesis
        available_kpis: KPI names from reference data
        available_experiment_types: Type titles from reference data

    Returns:
        HypothesisParserOutput or raises after max retries
    """
    last_error = None
    for attempt in range(MAX_SPECIALIST_RETRIES):
        try:
            result = await parse_hypothesis(
                hypothesis_text=hypothesis_text,
                available_kpis=available_kpis,
                available_experiment_types=available_experiment_types,
            )
            if result.confidence > 0.0 or result.primary_kpi:
                return result
            logger.warning(f"Hypothesis parse attempt {attempt+1}: low confidence, retrying")
        except Exception as e:
            last_error = e
            logger.warning(f"Hypothesis parse attempt {attempt+1} failed: {e}")

    if last_error:
        raise last_error
    return HypothesisParserOutput()


async def handoff_dates(
    user_input: str,
    current_date: str,
    flow_type: FlowType,
) -> DateValidatorOutput:
    """Route to date validator with retry logic.

    Args:
        user_input: User's date text
        current_date: Today YYYY-MM-DD
        flow_type: design or measure

    Returns:
        DateValidatorOutput or raises after max retries
    """
    last_error = None
    for attempt in range(MAX_SPECIALIST_RETRIES):
        try:
            result = await validate_dates(
                user_input=user_input,
                current_date=current_date,
                flow_type=flow_type,
            )
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Date validation attempt {attempt+1} failed: {e}")

    if last_error:
        raise last_error
    return DateValidatorOutput(is_valid=False, error="Date validation failed after retries")


async def handoff_columns(
    columns: List[ColumnMetadata],
    total_rows: int,
    user_kpi: str,
    flow_type: FlowType,
    preview_rows: Optional[List[Dict[str, Any]]] = None,
    id_columns: Optional[List[str]] = None,
) -> ColumnDetectorOutput:
    """Route to column detector with privacy masking and retry logic.

    Args:
        columns: Column metadata list
        total_rows: Total rows in dataset
        user_kpi: User's selected KPI name
        flow_type: design or measure
        preview_rows: Raw preview rows (will be masked before sending)
        id_columns: Known ID columns for masking

    Returns:
        ColumnDetectorOutput or raises after max retries
    """
    # Mask preview data before sending to GPT
    masked_preview = None
    if preview_rows:
        masked_preview = mask_preview_rows(
            rows=preview_rows,
            id_columns=id_columns,
            max_rows=5,
        )

    last_error = None
    for attempt in range(MAX_SPECIALIST_RETRIES):
        try:
            result = await detect_columns(
                columns=columns,
                total_rows=total_rows,
                user_kpi=user_kpi,
                flow_type=flow_type,
                masked_preview=masked_preview,
            )
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Column detection attempt {attempt+1} failed: {e}")

    if last_error:
        raise last_error
    return ColumnDetectorOutput()


async def handoff_update(
    current_experiment: Dict[str, Any],
    user_request: str,
    current_date: str,
) -> UpdateValidatorOutput:
    """Route to update validator with privacy masking and retry logic.

    Args:
        current_experiment: Full experiment JSON (will be masked)
        user_request: User's change request text
        current_date: Today YYYY-MM-DD

    Returns:
        UpdateValidatorOutput or raises after max retries
    """
    from src.modules.agent.utils.privacy import mask_experiment_data
    masked_experiment = mask_experiment_data(current_experiment)

    last_error = None
    for attempt in range(MAX_SPECIALIST_RETRIES):
        try:
            result = await validate_update(
                current_experiment=masked_experiment,
                user_request=user_request,
                current_date=current_date,
            )
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Update validation attempt {attempt+1} failed: {e}")

    if last_error:
        raise last_error
    return UpdateValidatorOutput(intent="cannot_update", warnings=["Validation failed after retries"])