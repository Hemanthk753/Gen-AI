"""Main orchestrator — routes messages across all 3 flows.

DESIGN flow (create experiment):
  Step 0: Greeting + flow detection
  Step 1: Hypothesis → parse
  Step 2: Dates → validate
  Step 3: Validation display → confirm
  Step 4: File upload prompt
  Step 5: Data processing + column detection
  Step 6: Column map confirm → save
  Step 7: Execute RCT/measurement → success

MEASUREMENT flow: Same as design but is_measurement_flow=True

UPDATE flow:
  Step 0: List experiments → user picks one
  Step 1: Show experiment details
  Step 2: Process change request → validate → apply
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional
import json
from openai import AsyncOpenAI
from src.modules.agent.config import (
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_MAX_TOKENS,
)
from src.modules.agent.prompts import ORCHESTRATOR_SYSTEM

_llm = AsyncOpenAI(api_key=OPENAI_API_KEY)
from src.modules.agent.config import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_CONTROL_ALLOCATION,
    DEFAULT_MDE_PERCENT,
    DEFAULT_POWER_OF_TEST,
    DEFAULT_TREATMENT_ALLOCATION,
)
from src.modules.agent.models.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentMessage,
    CollectedFields,
    ColumnMetadata,
    ConversationState,
    ExperimentCreatePayload,
    ExperimentUpdatePayload,
    FlowType,
    MessageType,
    SuggestedOption,
    UpdateIntent,
)
from src.modulesa.gent.services.api_client import APIClient
from src.modules.agent.services.context_loader import load_user_context
from src.modules.utils.handoff import handoff_hypothesis, handoff_dates, handoff_columns, handoff_update

logger = logging.getLogger(__name__)

# In-memory state store (replace with Redis/DB in production)
_states: dict[str, ConversationState] = {}


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════


async def _ask_orchestrator_llm(state: ConversationState, user_message: str) -> dict:
    """Call GPT for ambiguous routing, field corrections, and natural responses."""
    ctx = state.user_context
    profile = ctx.profile if ctx else None
    collected = state.collected

    system = ORCHESTRATOR_SYSTEM.format(
        user_name=profile.first_name if profile else "User",
        user_email=profile.email if profile else "",
        user_country=collected.country_id or "Not set",
        user_zone=collected.zone_literal or "Not set",
        user_bf=collected.business_function_id or "Not set",
        total_experiments=ctx.total_experiments if ctx else 0,
        most_used_use_case=ctx.most_used_use_case_title or "None",
        flow_type=state.flow_type.value if state.flow_type else "not_set",
        current_step=state.current_step,
        collected_fields_json=json.dumps(
            collected.model_dump(exclude_none=True, mode="json"),
            indent=2, default=str,
        ),
    )

    messages = [{"role": "system", "content": system}]
    for msg in state.messages[-10:]:
        messages.append({
            "role": "user" if msg.role == "user" else "assistant",
            "content": msg.content,
        })
    messages.append({"role": "user", "content": user_message})

    response = await _llm.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        max_tokens=OPENAI_MAX_TOKENS,
        messages=messages,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "continue", "message": raw}

async def handle_message(request: AgentChatRequest, auth_token: str) -> AgentChatResponse:
    """Main entry point. Routes each user message to the correct flow step."""
    conv_id = request.conversation_id or str(uuid.uuid4())
    state = _states.get(conv_id, ConversationState(conversation_id=conv_id))
    api = APIClient(auth_token=auth_token)

    state.messages.append(AgentMessage(
        role="user", content=request.message, timestamp=datetime.utcnow()
    ))

    if state.user_context is None:
        try:
            state.user_context = await load_user_context(api)
        except Exception as e:
            logger.error(f"Context load failed: {e}")
            return _error_response(conv_id, f"Failed to load your profile: {e}")

    try:
        if state.flow_type is None:
            response = await _handle_flow_detection(state, request.message)
        elif state.flow_type == FlowType.UPDATE:
            response = await _handle_update_flow(state, request.message, api)
        else:
            response = await _handle_create_flow(state, request.message, api)
    except Exception as e:
        logger.error(f"Orchestrator error: {e}", exc_info=True)
        response = _error_response(conv_id, f"Something went wrong: {e}")

    _states[conv_id] = state
    response.conversation_id = conv_id
    response.current_step = state.current_step
    response.flow_type = state.flow_type

    state.messages.append(AgentMessage(
        role="agent", content=response.message, timestamp=datetime.utcnow(),
        message_type=response.message_type,
    ))
    return response


# ═══════════════════════════════════════════════════════════════
# FLOW DETECTION (Step 0)
# ═══════════════════════════════════════════════════════════════


async def _handle_flow_detection(state: ConversationState, message: str) -> AgentChatResponse:
    msg_lower = message.lower()
    name = state.user_context.profile.first_name if state.user_context else "there"

    # Fast path: obvious keywords
    if any(kw in msg_lower for kw in ["update", "edit", "change", "modify"]):
        state.flow_type = FlowType.UPDATE
        state.current_step = 0
        return await _handle_update_flow(state, message, None)

    if any(kw in msg_lower for kw in ["measure", "measurement", "analyze past", "past experiment"]):
        state.flow_type = FlowType.MEASURE
        state.collected.flow_type = FlowType.MEASURE
        state.collected.is_measurement_flow = True
        state.current_step = 1
        return AgentChatResponse(
            message=(
                f"Hi {name}! 👋 Let's set up a **measurement** experiment.\n\n"
                "Please share your **hypothesis** — what do you believe happened "
                "and what KPI was impacted?\n\n"
                "_Example: 'We ran a 10% discount promotion in Q4 and believe it "
                "increased Net Revenue by 5%'_"
            ),
            conversation_id="", message_type=MessageType.TEXT,
        )

    if any(kw in msg_lower for kw in ["create", "new", "design", "test", "experiment"]):
        state.flow_type = FlowType.DESIGN
        state.collected.flow_type = FlowType.DESIGN
        state.collected.is_measurement_flow = False
        state.current_step = 1
        return AgentChatResponse(
            message=(
                f"Hi {name}! 👋 Let's create a new **experiment**.\n\n"
                "Please share your **hypothesis** — what do you want to test "
                "and what outcome do you expect?\n\n"
                "_Example: 'Offering a 10% discount to premium POCs will increase "
                "Net Revenue by 5%'_"
            ),
            conversation_id="", message_type=MessageType.TEXT,
        )

    # Fallback: GPT decides
    result = await _ask_orchestrator_llm(state, message)
    detected = result.get("detected_flow")

    if detected == "update":
        state.flow_type = FlowType.UPDATE
        state.current_step = 0
        return await _handle_update_flow(state, message, None)

    if detected == "measure":
        state.flow_type = FlowType.MEASURE
        state.collected.flow_type = FlowType.MEASURE
        state.collected.is_measurement_flow = True
        state.current_step = 1
        return AgentChatResponse(
            message=result.get("message", f"Hi {name}! Let's set up a measurement experiment.\n\nPlease share your hypothesis."),
            conversation_id="", message_type=MessageType.TEXT,
        )

    if detected == "question":
        return AgentChatResponse(
            message=result.get("message", "How can I help?"),
            conversation_id="", message_type=MessageType.TEXT,
            suggested_options=[
                SuggestedOption(label="🆕 Create experiment", value="create"),
                SuggestedOption(label="📊 Measure past experiment", value="measure"),
                SuggestedOption(label="✏️ Update experiment", value="update"),
            ],
        )

    # Default: design
    state.flow_type = FlowType.DESIGN
    state.collected.flow_type = FlowType.DESIGN
    state.collected.is_measurement_flow = False
    state.current_step = 1
    return AgentChatResponse(
        message=result.get("message", f"Hi {name}! Let's create a new experiment.\n\nPlease share your hypothesis."),
        conversation_id="", message_type=MessageType.TEXT,
    )
    

# ═══════════════════════════════════════════════════════════════
# CREATE FLOW (Design + Measurement) — Steps 1-7
# ═══════════════════════════════════════════════════════════════


async def _handle_create_flow(
    state: ConversationState, message: str, api: APIClient
) -> AgentChatResponse:
    step = state.current_step
    if step == 1:
        return await _step_hypothesis(state, message)
    elif step == 2:
        return await _step_dates(state, message)
    elif step == 3:
        return await _step_confirm_validation(state, message, api)
    elif step == 4:
        return await _step_upload_prompt(state)
    elif step == 5:
        return await _step_process_data(state, message, api)
    elif step == 6:
        return await _step_column_confirm(state, message, api)
    elif step == 7:
        return await _step_execute(state, api)
    else:
        return AgentChatResponse(
            message="Experiment is complete! 🎉 Start a new conversation to create another.",
            conversation_id="", message_type=MessageType.SUCCESS_SCREEN,
        )


async def _step_hypothesis(state: ConversationState, message: str) -> AgentChatResponse:
    """Step 1: Parse hypothesis via specialist."""
    ref = state.user_context.reference_data
    kpi_names = [k.name for k in ref.kpis]
    type_titles = [t.title for t in ref.experiment_types]

    result = await handoff_hypothesis(
    hypothesis_text=message,
    available_kpis=kpi_names,
    available_experiment_types=type_titles,
    )

    c = state.collected
    c.hypothesis = message
    c.primary_kpi = result.primary_kpi_match or result.primary_kpi
    c.primary_kpi_type = result.primary_kpi_type
    c.expected_uplift = result.expected_uplift
    c.experiment_type_hint = result.experiment_type_hint
    c.control_action = result.control_action
    c.treatment_action = result.treatment_action

    # Match KPI to literal from reference data
    for kpi in ref.kpis:
        if kpi.name == c.primary_kpi:
            c.primary_kpi_literal = kpi.kpi_literal
            c.primary_kpi_type = kpi.type.lower()
            break

    # Match experiment type to literal
    type_map = {"RCT": "rct", "MTE": "mte", "OS": "expty_003", "PrePost": "pre_post"}
    hint_val = result.experiment_type_hint.value if result.experiment_type_hint else "RCT"
    for et in ref.experiment_types:
        if hint_val.lower() in (et.title or "").lower() or \
           et.experiment_type_literal == type_map.get(hint_val):
            c.experiment_type_literal = et.experiment_type_literal
            break
    if not c.experiment_type_literal and ref.experiment_types:
        c.experiment_type_literal = ref.experiment_types[0].experiment_type_literal

    # Auto-fill from user profile
    profile = state.user_context.profile
    c.country_id = profile.primary_country_literal
    c.zone_literal = profile.primary_zone_literal
    c.business_function_id = profile.primary_business_function_literal
    c.currency_code = profile.primary_currency_code
    c.use_case_id = state.user_context.most_used_use_case_literal

    # Auto-generate name and problem statement
    kpi_part = (c.primary_kpi or "experiment")[:30]
    date_part = date.today().strftime("%b%Y")
    c.experiment_name = f"{kpi_part} - {hint_val} - {date_part}"
    c.problem_statement = f"Testing: {message[:200]}"

    state.current_step = 2

    return AgentChatResponse(
        message=(
            f"Great! I've understood your hypothesis:\n\n"
            f"📊 **KPI**: {c.primary_kpi or 'Not detected'}\n"
            f"📈 **Expected Uplift**: {c.expected_uplift or 'Not specified'}%\n"
            f"🧪 **Experiment Type**: {hint_val}\n"
            f"🎯 **Control**: {c.control_action or 'Business as usual'}\n"
            f"💊 **Treatment**: {c.treatment_action or 'Intervention applied'}\n\n"
            f"Now, what **dates** would you like for this experiment?\n\n"
            f"_Example: 'April 1 to June 30, 2026' or 'next 3 months'_"
        ),
        conversation_id="", message_type=MessageType.TEXT,
    )


async def _step_dates(state: ConversationState, message: str) -> AgentChatResponse:
    """Step 2: Validate dates via specialist."""
    today = date.today().isoformat()
    flow = state.collected.flow_type or FlowType.DESIGN

    result = await handoff_dates(user_input=message, current_date=today, flow_type=flow)

    if not result.is_valid:
        return AgentChatResponse(
            message=f"❌ {result.error or 'Invalid dates.'}\n\nPlease provide valid experiment dates.",
            conversation_id="", message_type=MessageType.ERROR,
        )

    c = state.collected
    c.start_date = result.start_date
    c.end_date = result.end_date
    c.duration_days = result.duration_days
    c.baseline_start = result.baseline_start
    c.baseline_end = result.baseline_end

    # Past dates in design flow → auto-switch to measurement
    if result.is_past and c.flow_type == FlowType.DESIGN:
        c.flow_type = FlowType.MEASURE
        c.is_measurement_flow = True
        state.flow_type = FlowType.MEASURE

    state.current_step = 3

    warnings_str = ""
    if result.warnings:
        warnings_str = "\n⚠️ " + "\n⚠️ ".join(result.warnings)
    if result.seasonality_alerts:
        warnings_str += "\n🗓️ " + "\n🗓️ ".join(result.seasonality_alerts)

    return AgentChatResponse(
        message=(
            f"Here's a summary of your experiment setup:\n\n"
            f"📝 **Name**: {c.experiment_name}\n"
            f"🔬 **Hypothesis**: {(c.hypothesis or '')[:100]}...\n"
            f"📊 **KPI**: {c.primary_kpi} ({c.primary_kpi_type})\n"
            f"📈 **Expected Uplift**: {c.expected_uplift or 'N/A'}%\n"
            f"🧪 **Type**: {c.experiment_type_hint.value if c.experiment_type_hint else 'RCT'}\n"
            f"📅 **Period**: {c.start_date} → {c.end_date} ({c.duration_days} days)\n"
            f"📅 **Baseline**: {c.baseline_start} → {c.baseline_end}\n"
            f"🌍 **Country**: {c.country_id}\n"
            f"🏢 **Business Function**: {c.business_function_id}\n"
            f"📋 **Use Case**: {c.use_case_id}"
            f"{warnings_str}\n\n"
            f"Does this look correct? Type **yes** to proceed or tell me what to change."
        ),
        conversation_id="", message_type=MessageType.VALIDATION_DISPLAY,
        suggested_options=[
            SuggestedOption(label="✅ Yes, proceed", value="yes"),
            SuggestedOption(label="✏️ Change something", value="change"),
        ],
    )


async def _step_confirm_validation(
    state: ConversationState, message: str, api: APIClient
) -> AgentChatResponse:
    """Step 3: User confirms → create experiment via POST /experiments."""
    if message.lower() not in ("yes", "confirm", "proceed", "ok", "y"):
        # GPT figures out what they want to change
        result = await _ask_orchestrator_llm(state, message)
        corrections = result.get("field_corrections", {})

        if "hypothesis" in corrections or "kpi" in corrections or "type" in corrections:
            state.current_step = 1
            return AgentChatResponse(
                message=result.get("message", "Sure! Please share your updated hypothesis."),
                conversation_id="", message_type=MessageType.TEXT,
            )
        if "date" in corrections or "start_date" in corrections or "end_date" in corrections:
            state.current_step = 2
            return AgentChatResponse(
                message=result.get("message", "Sure! Please provide the new dates."),
                conversation_id="", message_type=MessageType.TEXT,
            )

        return AgentChatResponse(
            message=result.get("message", "What would you like to change? Hypothesis, dates, KPI, or something else?"),
            conversation_id="", message_type=MessageType.TEXT,
            suggested_options=[
                SuggestedOption(label="🔬 Hypothesis/KPI", value="change hypothesis"),
                SuggestedOption(label="📅 Dates", value="change dates"),
                SuggestedOption(label="🔄 Start over", value="start over"),
            ],
        )

    # ── Everything below stays exactly the same ──
    c = state.collected
    metrices = []
    if c.primary_kpi_literal:
        metrices.append({
            "kpi_literal": c.primary_kpi_literal,
            "literral_value": c.primary_kpi,
            "kpi_type": c.primary_kpi_type,
        })

    success_criteria = []
    if c.primary_kpi_literal:
        success_criteria.append({
            "kpi_literal": c.primary_kpi_literal,
            "minimum_uplift": c.expected_uplift or DEFAULT_MDE_PERCENT,
            "confidence_level": c.confidence_level,
        })

    group_configs = [
        {"name": "Control", "action": c.control_action or "No change",
         "is_control": True, "group_allocation_percentage": DEFAULT_CONTROL_ALLOCATION},
        {"name": "Treatment", "action": c.treatment_action or "Intervention",
         "is_control": False, "group_allocation_percentage": DEFAULT_TREATMENT_ALLOCATION},
    ]

    payload = ExperimentCreatePayload(
        experiment_name=c.experiment_name or "Untitled Experiment",
        problem_statement=c.problem_statement,
        hypothesis=c.hypothesis,
        metrices=metrices,
        success_criteria=success_criteria,
        group_configs=group_configs,
        start_date=c.start_date,
        end_date=c.end_date,
        baseline_start_date=c.baseline_start,
        baseline_end_date=c.baseline_end,
        step=0,
        is_draft=True,
        is_measurement_flow=c.is_measurement_flow,
        experiment_type_id=c.experiment_type_literal,
        country_id=c.country_id,
        use_case_id=c.use_case_id,
        business_function_id=c.business_function_id,
        hs_itrvn_occured=c.is_measurement_flow or False,
    )

    result = await api.create_experiment(payload.model_dump(exclude_none=True, mode="json"))
    c.experiment_literal = result.get("experiment_literal", result.get("literal", ""))
    c.group_configs = group_configs

    state.current_step = 4
    return await _step_upload_prompt(state)


async def _step_upload_prompt(state: ConversationState) -> AgentChatResponse:
    """Step 4: Prompt user to upload data file."""
    c = state.collected
    upload_type = "measurement" if c.is_measurement_flow else "design"
    c.upload_type = upload_type

    return AgentChatResponse(
        message=(
            f"✅ Experiment **{c.experiment_name}** created successfully!\n\n"
            f"Now please upload your **{upload_type} data file** (CSV or Excel).\n\n"
            f"The file should contain your experiment data with columns for:\n"
            f"- **Granularity** (e.g., POC ID, store ID)\n"
            f"- **Date** column\n"
            f"- **{c.primary_kpi or 'KPI'}** metric column\n"
            f"{'- **Group** column (test/control assignment)' if c.is_measurement_flow else ''}"
        ),
        conversation_id="", message_type=MessageType.TEXT,
        requires_upload=True,
    )


async def _step_process_data(
    state: ConversationState, message: str, api: APIClient
) -> AgentChatResponse:
    """Step 5: After file upload — process data, detect columns.

    NOTE: File upload is handled by a separate endpoint. This step is
    triggered after the frontend confirms the upload is complete.
    The message contains the filename or "uploaded" confirmation.
    """
    c = state.collected
    literal = c.experiment_literal
    upload_type = c.upload_type or "design"

    # ── 5a. Trigger data processing (GET /experiments/process-validated-data) ──
    try:
        task_resp = await api.process_validated_data(literal, upload_type)
        task_id = task_resp.get("task_id", "")
        c.task_id = task_id
    except Exception as e:
        return _error_response(state.conversation_id, f"Data processing failed: {e}")

    # ── 5b. Poll until done ──
    poll_result = await api.poll_task_until_done(task_id)
    if poll_result.get("status") != "SUCCESS":
        return _error_response(
            state.conversation_id,
            f"Data processing failed: {poll_result.get('status', 'UNKNOWN')}"
        )

    # ── 5c. Get columns and preview (GET /experiments/get-columns-and-preview-data) ──
    table_type = "KPI" if c.is_measurement_flow else "Design"
    col_data = await api.get_columns_and_preview(literal, table_type=table_type)

    # Parse column metadata
    raw_columns = col_data.get("columns", col_data.get("columnns", []))
    total_rows = col_data.get("total_rows", 0)
    preview_rows = col_data.get("preview", col_data.get("data", {}).get("result", []))

    columns: List[ColumnMetadata] = []
    for rc in raw_columns:
        if isinstance(rc, dict):
            columns.append(ColumnMetadata(
                name=rc.get("col_name", rc.get("name", rc.get("column_name", ""))),
                data_type=rc.get("data_type", rc.get("type", "string")),
                cardinality=rc.get("cardinality", rc.get("distinct_count", 0)),
                null_pct=rc.get("null_pct", rc.get("null_percentage", 0.0)),
            ))

    c.total_rows = total_rows
    c.total_columns = len(columns)

    # ── 5d. Detect columns via specialist ──
    detection = await handoff_columns(
    columns=columns,
    total_rows=total_rows,
    user_kpi=c.primary_kpi or "",
    flow_type=c.flow_type or FlowType.DESIGN,
    preview_rows=preview_rows,
    id_columns=None,
    )

    # Build column map from detection
    col_map = {
        "granularities": detection.granularity.columns or (
            [detection.granularity.column] if detection.granularity.column else []
        ),
        "date": detection.date.column,
        "date_frequency": detection.date_frequency,
        "kpi": detection.kpi.column,
    }
    if c.is_measurement_flow and detection.group.column:
        col_map["group"] = detection.group.column

    if c.is_measurement_flow:
        c.measurement_column_map = col_map
    else:
        c.design_column_map = col_map

    state.current_step = 6

    # Format detection result for user confirmation
    gran_display = ", ".join(col_map.get("granularities", []))
    features_display = ", ".join(detection.features[:5]) if detection.features else "None detected"
    blocking_display = ", ".join(detection.blocking_factors[:5]) if detection.blocking_factors else "None detected"

    return AgentChatResponse(
        message=(
            f"📊 Data processed! **{total_rows:,}** rows, **{len(columns)}** columns.\n\n"
            f"Here's what I detected:\n\n"
            f"🔑 **Granularity**: {gran_display}\n"
            f"📅 **Date Column**: {col_map.get('date', 'Not found')}\n"
            f"📊 **KPI Column**: {col_map.get('kpi', 'Not found')}\n"
            f"{'👥 **Group Column**: ' + col_map.get('group', 'Not found') if c.is_measurement_flow else ''}\n"
            f"📐 **Features**: {features_display}\n"
            f"🧱 **Blocking Factors**: {blocking_display}\n\n"
            f"Does this mapping look correct? Type **yes** to proceed or specify corrections."
        ),
        conversation_id="", message_type=MessageType.COLUMN_DETECTION,
        suggested_options=[
            SuggestedOption(label="✅ Yes, proceed", value="yes"),
            SuggestedOption(label="✏️ Change mapping", value="change"),
        ],
    )


async def _step_column_confirm(
    state: ConversationState, message: str, api: APIClient
) -> AgentChatResponse:
    """Step 6: User confirms column mapping → save via PUT /experiments/{literal}."""
    if message.lower() not in ("yes", "confirm", "proceed", "ok", "y"):
        # GPT parses column corrections
        result = await _ask_orchestrator_llm(state, message)
        corrections = result.get("field_corrections", {})

        if not corrections:
            return AgentChatResponse(
                message=result.get("message", "Please specify which column to change."),
                conversation_id="", message_type=MessageType.TEXT,
            )

        c = state.collected
        col_map = (c.measurement_column_map if c.is_measurement_flow else c.design_column_map) or {}

        for role, value in corrections.items():
            if role in ("granularity", "granularities"):
                col_map["granularities"] = value if isinstance(value, list) else [value]
            elif role in ("date", "date_column"):
                col_map["date"] = value
            elif role in ("kpi", "kpi_column"):
                col_map["kpi"] = value
            elif role in ("group", "group_column"):
                col_map["group"] = value

        if c.is_measurement_flow:
            c.measurement_column_map = col_map
        else:
            c.design_column_map = col_map

        gran_display = ", ".join(col_map.get("granularities", []))
        return AgentChatResponse(
            message=(
                f"Updated:\n\n"
                f"🔑 **Granularity**: {gran_display}\n"
                f"📅 **Date**: {col_map.get('date', 'Not set')}\n"
                f"📊 **KPI**: {col_map.get('kpi', 'Not set')}\n"
                f"{'👥 **Group**: ' + col_map.get('group', 'Not set') if c.is_measurement_flow else ''}\n\n"
                f"Correct now?"
            ),
            conversation_id="", message_type=MessageType.COLUMN_DETECTION,
            suggested_options=[
                SuggestedOption(label="✅ Yes", value="yes"),
                SuggestedOption(label="✏️ Change more", value="change"),
            ],
        )

    # ── Everything below stays exactly the same ──
    c = state.collected
    literal = c.experiment_literal

    update_data: Dict = {"step": 2}

    if c.is_measurement_flow:
        update_data["measurement_column_map"] = c.measurement_column_map
        update_data["measurement_data_quality_report"] = c.data_quality_report or {}
    else:
        update_data["design_column_map"] = c.design_column_map
        update_data["design_data_quality_report"] = c.data_quality_report or {}
        update_data["design_blocking_factors_overwrite"] = []
        update_data["design_table_filters"] = {"filters": {}}

    update_data["group_configs"] = c.group_configs
    update_data["total_group_size"] = c.total_rows or 0
    update_data["metrices"] = []
    if c.primary_kpi_literal:
        update_data["metrices"] = [{
            "kpi_literal": c.primary_kpi_literal,
            "literral_value": c.primary_kpi,
            "kpi_type": c.primary_kpi_type,
        }]

    if c.file_name:
        update_data["file_name"] = c.file_name
    if c.design_table_name:
        update_data["design_table_name"] = c.design_table_name

    await api.update_experiment(literal, update_data)

    try:
        kpi_stats = await api.get_design_kpi_measurement(literal)
        avg_baseline = kpi_stats.get("mean", 0)
        std_dev = kpi_stats.get("deviation", 0)

        if avg_baseline and std_dev:
            estimation_type = c.primary_kpi_type or "continuous"
            sample_payload = {
                "estimation_type": estimation_type,
                "mde": (c.expected_uplift or DEFAULT_MDE_PERCENT) / 100 * avg_baseline,
                "confidence_level": c.confidence_level / 100,
                "avg_baseline": avg_baseline,
                "std_deviation": std_dev,
                "power_of_test": DEFAULT_POWER_OF_TEST / 100,
            }
            sample_result = await api.estimate_sample_size(sample_payload)
            c.smaller_sample_size = sample_result.get("min_control_sample_size", 0)
            c.recommended_sample_size = sample_result.get("min_test_sample_size", 0)

            await api.update_experiment(literal, {
                "smaller_sample_size": c.smaller_sample_size,
                "recommended_sample_size": c.recommended_sample_size,
                "step": 3,
            })
    except Exception as e:
        logger.warning(f"Sample size estimation skipped: {e}")

    state.current_step = 7
    return await _step_execute(state, api)


async def _step_execute(state: ConversationState, api: APIClient) -> AgentChatResponse:
    """Step 7: Execute RCT or measurement analysis → show success."""
    c = state.collected
    literal = c.experiment_literal

    if c.is_measurement_flow:
        # ── Measurement: run-aks-jobs ──
        try:
            task_resp = await api.run_aks_jobs(literal)
            task_id = task_resp.get("task_id", "")
            poll_result = await api.poll_task_until_done(task_id)

            if poll_result.get("status") != "SUCCESS":
                return _error_response(
                    state.conversation_id,
                    f"Measurement analysis failed: {poll_result.get('status', 'UNKNOWN')}"
                )

            # Get merge overview
            merge_data = await api.get_merge_overview(literal)
            await api.update_experiment(literal, {"step": 6, "is_draft": False})

            state.is_complete = True
            state.current_step = 8

            return AgentChatResponse(
                message=(
                    f"🎉 **Measurement analysis complete!**\n\n"
                    f"📊 Experiment: **{c.experiment_name}**\n"
                    f"📈 KPI: {c.primary_kpi}\n"
                    f"📅 Period: {c.start_date} → {c.end_date}\n\n"
                    f"Your results are ready for review in the TestOps dashboard."
                ),
                conversation_id="",
                message_type=MessageType.SUCCESS_SCREEN,
                metadata={"experiment_literal": literal, "merge_overview": merge_data},
            )
        except Exception as e:
            return _error_response(state.conversation_id, f"Measurement execution failed: {e}")

    else:
        # ── Design: execute RCT ──
        try:
            task_resp = await api.execute_rct(literal)
            task_id = task_resp.get("task_id", "")
            poll_result = await api.poll_task_until_done(task_id)

            if poll_result.get("status") != "SUCCESS":
                return _error_response(
                    state.conversation_id,
                    f"RCT execution failed: {poll_result.get('status', 'UNKNOWN')}"
                )

            # Get RCT results + download link
            rct_result = await api.get_rct_result(literal)
            download_resp = await api.get_rct_download_link(literal)
            download_link = download_resp.get("download_link", "")

            await api.update_experiment(literal, {"step": 6, "is_draft": False})

            state.is_complete = True
            state.current_step = 8

            return AgentChatResponse(
                message=(
                    f"🎉 **RCT design complete!**\n\n"
                    f"📊 Experiment: **{c.experiment_name}**\n"
                    f"📈 KPI: {c.primary_kpi}\n"
                    f"📅 Period: {c.start_date} → {c.end_date}\n"
                    f"👥 Total Scope: {c.total_rows:,} units\n"
                    f"📉 Control: {c.control_allocation}% | Treatment: {c.treatment_allocation}%\n"
                    f"{'📏 Min Sample Size: ' + str(c.recommended_sample_size) if c.recommended_sample_size else ''}\n\n"
                    f"{'📥 [Download RCT Output](' + download_link + ')' if download_link else ''}\n\n"
                    f"Your experiment groups have been assigned!"
                ),
                conversation_id="",
                message_type=MessageType.SUCCESS_SCREEN,
                metadata={
                    "experiment_literal": literal,
                    "download_link": download_link,
                    "rct_result": rct_result,
                },
            )
        except Exception as e:
            return _error_response(state.conversation_id, f"RCT execution failed: {e}")


# ═══════════════════════════════════════════════════════════════
# UPDATE FLOW — Steps 0-2
# ═══════════════════════════════════════════════════════════════


async def _handle_update_flow(
    state: ConversationState, message: str, api: Optional[APIClient]
) -> AgentChatResponse:
    step = state.current_step

    if step == 0:
        return await _update_list_experiments(state)
    elif step == 1:
        return await _update_select_experiment(state, message, api)
    elif step == 2:
        return await _update_process_change(state, message, api)
    else:
        return AgentChatResponse(
            message="Update complete! Start a new conversation for more changes.",
            conversation_id="", message_type=MessageType.TEXT,
        )


async def _update_list_experiments(state: ConversationState) -> AgentChatResponse:
    """Update Step 0: Show user's experiments list."""
    experiments = state.user_context.experiments
    state.experiments_list = experiments

    if not experiments:
        state.flow_type = None
        state.current_step = 0
        return AgentChatResponse(
            message=(
                "You don't have any experiments yet. "
                "Would you like to create a new one instead?"
            ),
            conversation_id="", message_type=MessageType.TEXT,
            suggested_options=[
                SuggestedOption(label="🆕 Create new experiment", value="create"),
            ],
        )

    exp_lines = []
    for i, exp in enumerate(experiments[:20], 1):
        status_text = exp.status_text or "Draft"
        exp_lines.append(f"{i}. **{exp.experiment_name or exp.experiment_literal}** — {status_text}")

    exp_list_str = "\n".join(exp_lines)
    state.current_step = 1

    return AgentChatResponse(
        message=(
            f"Here are your experiments:\n\n{exp_list_str}\n\n"
            f"Which experiment would you like to update? "
            f"Type the **number** or **name**."
        ),
        conversation_id="",
        message_type=MessageType.EXPERIMENT_LIST,
        suggested_options=[
            SuggestedOption(label=exp.experiment_name or exp.experiment_literal,
                          value=exp.experiment_literal)
            for exp in experiments[:10]
        ],
    )


async def _update_select_experiment(
    state: ConversationState, message: str, api: APIClient
) -> AgentChatResponse:
    """Update Step 1: User selects experiment → fetch full details → show."""
    experiments = state.experiments_list or state.user_context.experiments
    selected = None

    # Match by number
    try:
        idx = int(message.strip()) - 1
        if 0 <= idx < len(experiments):
            selected = experiments[idx]
    except ValueError:
        pass

    # Match by literal
    if not selected:
        for exp in experiments:
            if exp.experiment_literal == message.strip():
                selected = exp
                break

    # Match by name (fuzzy)
    if not selected:
        msg_lower = message.lower().strip()
        for exp in experiments:
            if msg_lower in (exp.experiment_name or "").lower():
                selected = exp
                break

    if not selected:
        return AgentChatResponse(
            message="I couldn't find that experiment. Please try again with the number or name.",
            conversation_id="", message_type=MessageType.TEXT,
        )

    # Fetch full experiment details
    state.update_experiment_literal = selected.experiment_literal
    full_exp = await api.get_experiment(selected.experiment_literal)
    state.update_experiment_data = full_exp

    # Display current details
    exp_name = full_exp.get("experiment_name", "")
    hypothesis = full_exp.get("hypothesis", "N/A")
    status_text = full_exp.get("status_text", "Draft")
    exp_type = full_exp.get("experiment_type", {})
    country = full_exp.get("country", {})
    start = full_exp.get("start_date", "N/A")
    end = full_exp.get("end_date", "N/A")
    kpi = full_exp.get("default_kpi", "N/A")

    state.current_step = 2

    return AgentChatResponse(
        message=(
            f"📋 **{exp_name}**\n\n"
            f"🔬 **Hypothesis**: {hypothesis}\n"
            f"📊 **KPI**: {kpi}\n"
            f"🧪 **Type**: {exp_type.get('title', 'N/A') if isinstance(exp_type, dict) else exp_type}\n"
            f"🌍 **Country**: {country.get('title', 'N/A') if isinstance(country, dict) else country}\n"
            f"📅 **Period**: {start} → {end}\n"
            f"📌 **Status**: {status_text}\n\n"
            f"What would you like to change?"
        ),
        conversation_id="", message_type=MessageType.EXPERIMENT_DETAILS,
    )


async def _update_process_change(
    state: ConversationState, message: str, api: APIClient
) -> AgentChatResponse:
    """Update Step 2: Validate change request → apply update."""
    today = date.today().isoformat()

    result = await handoff_update(
        current_experiment=state.update_experiment_data,
        user_request=message,
        current_date=today,
    )

    if result.intent == UpdateIntent.CANNOT_UPDATE:
        blocked_str = ", ".join(result.blocked_fields) if result.blocked_fields else "these fields"
        return AgentChatResponse(
            message=(
                f"❌ This change cannot be applied.\n\n"
                f"**Reason**: {result.warnings[0] if result.warnings else 'Experiment status does not allow this change.'}\n"
                f"**Blocked fields**: {blocked_str}"
            ),
            conversation_id="", message_type=MessageType.ERROR,
        )

    if result.confirmation_needed:
        # TODO: Implement confirmation flow for destructive changes
        pass

    # Apply update via PUT /experiments/{literal}
    literal = state.update_experiment_literal
    update_payload = {**result.update_payload, **result.cascade_updates}

    if result.requires_reprocessing and result.reprocessing_from_step is not None:
        update_payload["step"] = result.reprocessing_from_step

    if update_payload:
        await api.update_experiment(literal, update_payload)

    # Refresh experiment data
    state.update_experiment_data = await api.get_experiment(literal)

    warnings_str = ""
    if result.warnings:
        warnings_str = "\n⚠️ " + "\n⚠️ ".join(result.warnings)

    field_display = result.primary_field or "requested fields"

    if result.requires_reprocessing:
        return AgentChatResponse(
            message=(
                f"✅ **{field_display}** updated successfully!{warnings_str}\n\n"
                f"⚠️ This change requires re-processing from step {result.reprocessing_from_step}. "
                f"Please re-upload your data file."
            ),
            conversation_id="", message_type=MessageType.TEXT,
            requires_upload=True,
        )

    return AgentChatResponse(
        message=(
            f"✅ **{field_display}** updated successfully!{warnings_str}\n\n"
            f"Would you like to make any other changes?"
        ),
        conversation_id="", message_type=MessageType.TEXT,
        suggested_options=[
            SuggestedOption(label="✏️ Make another change", value="change"),
            SuggestedOption(label="✅ Done", value="done"),
        ],
    )


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════


def _error_response(conv_id: str, message: str) -> AgentChatResponse:
    """Build a standardized error response."""
    return AgentChatResponse(
        message=f"❌ {message}\n\nPlease try again or type 'start over' to restart.",
        conversation_id=conv_id,
        message_type=MessageType.ERROR,
    )