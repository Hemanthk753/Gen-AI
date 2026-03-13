"""Specialist: Parses hypothesis text into structured experiment parameters.

Input:  hypothesis_text + available KPIs + available experiment types
Output: HypothesisParserOutput (via GPT structured output)
"""

from __future__ import annotations

import json
import logging
from typing import List

from openai import AsyncOpenAI

from src.modules.agent.config import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from src.modules.agent.models.schemas import HypothesisParserOutput
from src.modules.agent.prompts import HYPOTHESIS_PARSER_SYSTEM

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def parse_hypothesis(
    hypothesis_text: str,
    available_kpis: List[str],
    available_experiment_types: List[str],
    actions_text: str | None = None,
) -> HypothesisParserOutput:
    """Parse a hypothesis into structured experiment parameters.

    Args:
        hypothesis_text: User's hypothesis (e.g., "Offering 10% discount will increase Net Revenue by 5%")
        available_kpis: List of KPI names from reference data (e.g., ["Net Revenue", "Volume"])
        available_experiment_types: List of type titles (e.g., ["RCT", "Observational Study"])
        actions_text: Optional additional actions description

    Returns:
        HypothesisParserOutput with extracted fields
    """
    user_message = f"""Hypothesis: {hypothesis_text}

Available KPIs: {json.dumps(available_kpis)}
Available Experiment Types: {json.dumps(available_experiment_types)}"""

    if actions_text:
        user_message += f"\n\nAdditional actions/context: {actions_text}"

    response = await client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": HYPOTHESIS_PARSER_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        response_format=HypothesisParserOutput,
    )

    result = response.choices[0].message.parsed
    logger.info(
        f"Hypothesis parsed: kpi={result.primary_kpi_match}, "
        f"type={result.experiment_type_hint}, "
        f"uplift={result.expected_uplift}, "
        f"confidence={result.confidence}"
    )
    return result