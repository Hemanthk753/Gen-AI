"""Loads user profile, reference data, and experiment history at conversation start.

Builds UserContext with derived fields for auto-fill during experiment creation.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from src.modules.agent.models.schemas import (
    BusinessFunctionRef,
    CountryRef,
    ExperimentHistoryItem,
    ExperimentTypeRef,
    KpiRef,
    ReferenceData,
    UserContext,
    UserProfile,
    UseCaseRef,
    ZoneRef,
)
from src.modules.agent.services.api_client import APIClient

logger = logging.getLogger(__name__)


async def load_user_context(api: APIClient) -> UserContext:
    """Load complete user context from backend APIs.

    Calls:
      GET /auth/me → user_literal
      GET /users/{literal} → full profile with zones/countries/business_functions
      GET /experiment-types, /kpis, /countries, /zones, /business-functions, /use-cases
      POST /experiments/get-all-experiments → user's experiment history
    """

    # ── 1. User Profile ──────────────────────────────────────────
    me = await api.get_current_user()
    user_literal = me.get("user_literal", me.get("literal", ""))
    user_detail = await api.get_user_details(user_literal) if user_literal else me

    profile = UserProfile(
        id=user_detail.get("id", 0),
        user_literal=user_literal,
        first_name=user_detail.get("first_name", ""),
        last_name=user_detail.get("last_name"),
        email=user_detail.get("email", ""),
        zones=_parse_zones(user_detail.get("zones", [])),
        countries=_parse_countries(user_detail.get("countries", [])),
        business_functions=_parse_bfs(user_detail.get("business_functions", [])),
    )

    # Derive primary values from first items
    if profile.countries:
        profile.primary_country_literal = profile.countries[0].country_literal
        profile.primary_currency_code = profile.countries[0].currency_code
    if profile.zones:
        profile.primary_zone_literal = profile.zones[0].zone_literal
    if profile.business_functions:
        profile.primary_business_function_literal = profile.business_functions[0].business_function_literal

    # ── 2. Reference Data ────────────────────────────────────────
    exp_types_raw, kpis_raw, countries_raw, zones_raw, bfs_raw, ucs_raw = (
        await api.list_experiment_types(),
        await api.list_kpis(),
        await api.list_countries(),
        await api.list_zones(),
        await api.list_business_functions(),
        await api.list_use_cases(),
    )

    ref = ReferenceData(
        experiment_types=[ExperimentTypeRef(**et) for et in _safe_list(exp_types_raw)],
        kpis=[KpiRef(**k) for k in _safe_list(kpis_raw)],
        countries=_parse_countries(_safe_list(countries_raw)),
        zones=_parse_zones(_safe_list(zones_raw)),
        business_functions=_parse_bfs(_safe_list(bfs_raw)),
        use_cases=[
            UseCaseRef(
                title=uc.get("title", ""),
                use_case_literal=uc.get("use_case_literal"),
                business_functions=_parse_bfs(uc.get("business_functions", [])),
            )
            for uc in _safe_list(ucs_raw)
        ],
    )

    # ── 3. Experiment History ────────────────────────────────────
    exp_list_resp = await api.list_experiments(
        filters={"only_my_experiments": True}, per_page=200
    )
    raw_experiments = _safe_list(
        exp_list_resp.get("result", exp_list_resp) if isinstance(exp_list_resp, dict) else exp_list_resp
    )

    experiments = []
    for exp in raw_experiments:
        try:
            experiments.append(ExperimentHistoryItem(
                experiment_literal=exp.get("experiment_literal", ""),
                experiment_name=exp.get("experiment_name"),
                status=exp.get("status"),
                status_text=exp.get("status_text"),
                step=exp.get("step"),
                step_text=exp.get("step_text"),
                experiment_type=exp.get("experiment_type"),
                country=exp.get("country"),
                use_case=exp.get("use_case"),
                business_function=exp.get("business_function"),
                default_kpi=exp.get("default_kpi"),
                uplift=exp.get("uplift"),
                start_date=exp.get("start_date"),
                end_date=exp.get("end_date"),
                is_measurement_flow=exp.get("is_measurement_flow"),
                is_draft=exp.get("is_draft"),
                created_at=exp.get("created_at"),
            ))
        except Exception as e:
            logger.warning(f"Skipping experiment parse error: {e}")

    # ── 4. Compute History Stats ─────────────────────────────────
    ctx = UserContext(
        profile=profile,
        experiments=experiments,
        reference_data=ref,
        total_experiments=len(experiments),
    )

    # Most used use case
    uc_counter: Counter = Counter()
    for exp in experiments:
        uc = exp.use_case
        if uc and isinstance(uc, dict) and uc.get("use_case_literal"):
            uc_counter[uc["use_case_literal"]] += 1
    if uc_counter:
        top_uc_literal, _ = uc_counter.most_common(1)[0]
        ctx.most_used_use_case_literal = top_uc_literal
        for uc_ref in ref.use_cases:
            if uc_ref.use_case_literal == top_uc_literal:
                ctx.most_used_use_case_title = uc_ref.title
                break

    # Average duration
    durations = []
    for exp in experiments:
        if exp.start_date and exp.end_date:
            delta = (exp.end_date - exp.start_date).days
            if delta > 0:
                durations.append(delta)
    if durations:
        ctx.average_duration_days = sum(durations) / len(durations)

    logger.info(
        f"Context loaded: user={profile.first_name}, "
        f"experiments={len(experiments)}, "
        f"kpis={len(ref.kpis)}, types={len(ref.experiment_types)}"
    )
    return ctx


# ── Helpers ──────────────────────────────────────────────────────


def _safe_list(data) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("result", [])
    return []


def _parse_zones(raw: list) -> list[ZoneRef]:
    return [
        ZoneRef(id=z.get("id", 0), title=z.get("title", ""), zone_literal=z.get("zone_literal"))
        for z in raw if isinstance(z, dict)
    ]


def _parse_countries(raw: list) -> list[CountryRef]:
    return [
        CountryRef(
            id=c.get("id", 0), title=c.get("title", ""),
            country_code=c.get("country_code", ""),
            currency_code=c.get("currency_code", ""),
            country_literal=c.get("country_literal"),
            zone=ZoneRef(**c["zone"]) if isinstance(c.get("zone"), dict) else None,
        )
        for c in raw if isinstance(c, dict)
    ]


def _parse_bfs(raw: list) -> list[BusinessFunctionRef]:
    return [
        BusinessFunctionRef(
            id=b.get("id", 0), title=b.get("title", ""),
            business_function_literal=b.get("business_function_literal"),
        )
        for b in raw if isinstance(b, dict)
    ]