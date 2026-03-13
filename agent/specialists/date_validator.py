"""Specialist: Validates and parses date inputs for experiment timelines.

Input:  user date text + current date + flow type
Output: DateValidatorOutput (via GPT structured output)
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from src.modules.agent.config import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from src.modules.agent.models.schemas import DateValidatorOutput, FlowType
from src.modules.agent.prompts import DATE_VALIDATOR_SYSTEM

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def validate_dates(
    user_input: str,
    current_date: str,
    flow_type: FlowType,
) -> DateValidatorOutput:
    """Validate and parse user date inputs.

    Args:
        user_input: Natural language date input (e.g., "next 3 months", "Jan 15 to April 15 2026")
        current_date: Today's date in YYYY-MM-DD format
        flow_type: FlowType.DESIGN or FlowType.MEASURE

    Returns:
        DateValidatorOutput with parsed dates, baseline, warnings
    """
    system_prompt = DATE_VALIDATOR_SYSTEM.format(
        current_date=current_date,
        flow_type=flow_type.value,
    )

    user_message = f"User's date input: {user_input}"

    response = await client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=DateValidatorOutput,
    )

    result = response.choices[0].message.parsed
    logger.info(
        f"Dates validated: {result.start_date} → {result.end_date}, "
        f"duration={result.duration_days}, valid={result.is_valid}, past={result.is_past}"
    )
    return result