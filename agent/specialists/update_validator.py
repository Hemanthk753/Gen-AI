"""Specialist: Validates experiment update requests and produces update plans.

Input:  current experiment data + user change request + experiment status
Output: UpdateValidatorOutput (via GPT structured output)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from openai import AsyncOpenAI

from src.modules.agent.config import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from src.modules.agent.models.schemas import UpdateValidatorOutput
from src.modules.agent.prompts import UPDATE_VALIDATOR_SYSTEM

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def validate_update(
    current_experiment: Dict[str, Any],
    user_request: str,
    current_date: str,
) -> UpdateValidatorOutput:
    """Validate an experiment update request.

    Args:
        current_experiment: Full experiment JSON from GET /experiments/{literal}
        user_request: User's change request text (e.g., "change the KPI to Volume")
        current_date: Today's date in YYYY-MM-DD format

    Returns:
        UpdateValidatorOutput with intent, update_payload, cascade_updates, warnings
    """
    # Trim experiment JSON to relevant fields to stay within token limits
    trimmed = _trim_experiment(current_experiment)

    system_prompt = UPDATE_VALIDATOR_SYSTEM.format(
        experiment_json=json.dumps(trimmed, indent=2, default=str),
        current_date=current_date,
    )

    user_message = f"Change request: {user_request}"

    response = await client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=UpdateValidatorOutput,
    )

    result = response.choices[0].message.parsed
    logger.info(
        f"Update validated: intent={result.intent}, "
        f"field={result.primary_field}, "
        f"reprocess={result.requires_reprocessing}, "
        f"warnings={len(result.warnings)}"
    )
    return result


def _trim_experiment(exp: Dict) -> Dict:
    """Keep only fields relevant for update validation to save tokens."""
    keep_keys = [
        "experiment_literal", "experiment_name", "hypothesis", "problem_statement",
        "status", "status_text", "step", "step_text",
        "start_date", "end_date", "baseline_start_date", "baseline_end_date",
        "is_measurement_flow", "is_draft",
        "experiment_type", "country", "use_case", "business_function",
        "default_kpi", "metrices", "success_criteria",
        "group_configs", "interventions", "exp_interventions",
        "smaller_sample_size", "recommended_sample_size",
    ]
    return {k: exp[k] for k in keep_keys if k in exp}