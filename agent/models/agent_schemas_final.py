"""Agent schemas — single source of truth for all data shapes.

Schema-driven: These Pydantic models are passed to OpenAI's structured
output (response_format) so GPT returns valid JSON matching these shapes.

Cross-checked against:
  - src/modules/experiments/schemas/common.py
  - src/modules/experiments/schemas/experience_metadatas.py
  - src/modules/experiments/schemas/sample_size.py
  - src/modules/users/schemas/common.py
  - src/modules/countries/schemas/common.py
  - src/modules/zones/schemas/common.py
  - src/modules/kpis/schemas/common.py
  - src/modules/use_cases/schemas/common.py
  - src/modules/business_functions/schemas/common.py
  - src/modules/experiment_types/schemas/common.py
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────
# ENUMS
# ────────────────────────────────────────────────────────────────


class FlowType(str, Enum):
    """Agent conversation flow type."""
    DESIGN = "design"
    MEASURE = "measure"
    UPDATE = "update"


class ExperimentTypeHint(str, Enum):
    """Experiment type detected from hypothesis."""
    RCT = "RCT"
    MTE = "MTE"
    OS = "OS"
    PRE_POST = "PrePost"


class MessageType(str, Enum):
    """Types of messages the agent can send to the frontend."""
    TEXT = "text"
    VALIDATION_DISPLAY = "validation_display"
    EXPERIMENT_DETAILS = "experiment_details"
    EXPERIMENT_LIST = "experiment_list"
    SUCCESS_SCREEN = "success_screen"
    OPTIONS = "options"
    COLUMN_DETECTION = "column_detection"
    ERROR = "error"


class UpdateIntent(str, Enum):
    """What the update validator determined."""
    UPDATE_FIELD = "update_field"
    RE_UPLOAD_DATA = "re_upload_data"
    SWITCH_TYPE = "switch_type"
    CANNOT_UPDATE = "cannot_update"


# ────────────────────────────────────────────────────────────────
# REFERENCE DATA
# Loaded once at conversation start via GET endpoints.
# Agent uses these to validate user inputs and auto-fill.
# ────────────────────────────────────────────────────────────────


class ZoneRef(BaseModel):
    """GET /v1/zones → each zone item"""
    id: int
    title: str
    zone_literal: Optional[str] = None


class CountryRef(BaseModel):
    """GET /v1/countries → each country item"""
    id: int
    title: str
    country_code: str
    currency_code: str
    country_literal: Optional[str] = None
    zone: Optional[ZoneRef] = None


class BusinessFunctionRef(BaseModel):
    """GET /v1/business-functions → each item"""
    id: int
    title: str
    business_function_literal: Optional[str] = None


class UseCaseRef(BaseModel):
    """GET /v1/use-cases → each item"""
    title: str
    use_case_literal: Optional[str] = None
    business_functions: Optional[List[BusinessFunctionRef]] = None


class ExperimentTypeRef(BaseModel):
    """GET /v1/experiment-types → each item"""
    title: str
    experiment_type_literal: Optional[str] = None


class KpiRef(BaseModel):
    """GET /v1/kpis → each item"""
    name: str
    type: str  # "Continuous" or "Proportion"
    kpi_literal: Optional[str] = None


class ReferenceData(BaseModel):
    """All master data fetched once at conversation start."""
    experiment_types: List[ExperimentTypeRef] = []
    kpis: List[KpiRef] = []
    countries: List[CountryRef] = []
    zones: List[ZoneRef] = []
    business_functions: List[BusinessFunctionRef] = []
    use_cases: List[UseCaseRef] = []


# ────────────────────────────────────────────────────────────────
# USER CONTEXT
# Built from auth + user + experiment list APIs.
# ────────────────────────────────────────────────────────────────


class UserProfile(BaseModel):
    """From GET /auth/me + GET /users/{literal}"""
    id: int
    user_literal: Optional[str] = None
    first_name: str
    last_name: Optional[str] = None
    email: str
    zones: Optional[List[ZoneRef]] = None
    countries: Optional[List[CountryRef]] = None
    business_functions: Optional[List[BusinessFunctionRef]] = None

    # Derived by agent from first item in lists
    primary_country_literal: Optional[str] = None
    primary_zone_literal: Optional[str] = None
    primary_business_function_literal: Optional[str] = None
    primary_currency_code: Optional[str] = None


class ExperimentHistoryItem(BaseModel):
    """Single experiment from POST /experiments/get-all-experiments.
    Used for: experiment list display + use case scoring + update flow."""
    experiment_literal: str
    experiment_name: Optional[str] = None
    status: Optional[Any] = None
    status_text: Optional[str] = None
    step: Optional[int] = None
    step_text: Optional[str] = None
    experiment_type: Optional[Dict] = None        # {title, experiment_type_literal}
    country: Optional[Dict] = None                 # {title, country_literal}
    use_case: Optional[Dict] = None                # {title, use_case_literal}
    business_function: Optional[Dict] = None       # {title, business_function_literal}
    default_kpi: Optional[str] = None
    uplift: Optional[float] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_measurement_flow: Optional[bool] = None
    is_draft: Optional[bool] = None
    created_at: Optional[datetime] = None


class UserContext(BaseModel):
    """Complete user context loaded at conversation start."""
    profile: UserProfile
    experiments: List[ExperimentHistoryItem] = []
    reference_data: ReferenceData = ReferenceData()

    # Computed from experiments list (for use case auto-detection)
    most_used_use_case_literal: Optional[str] = None
    most_used_use_case_title: Optional[str] = None
    average_duration_days: Optional[float] = None
    total_experiments: int = 0


# ────────────────────────────────────────────────────────────────
# SPECIALIST OUTPUTS
# Passed to OpenAI's response_format → GPT forced to match shape.
# ────────────────────────────────────────────────────────────────


class HypothesisParserOutput(BaseModel):
    """GPT structured output: hypothesis → structured fields."""
    primary_kpi: Optional[str] = None
    primary_kpi_match: Optional[str] = None
    primary_kpi_literal: Optional[str] = None
    primary_kpi_type: Optional[str] = None         # "continuous" or "proportion"
    expected_uplift: Optional[float] = None
    experiment_type_hint: ExperimentTypeHint = ExperimentTypeHint.RCT
    control_action: Optional[str] = None
    treatment_action: Optional[str] = None
    target_segment: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DateValidatorOutput(BaseModel):
    """GPT structured output: date text → validated dates."""
    start_date: Optional[str] = None               # YYYY-MM-DD
    end_date: Optional[str] = None                  # YYYY-MM-DD
    duration_days: Optional[int] = None
    baseline_start: Optional[str] = None            # YYYY-MM-DD
    baseline_end: Optional[str] = None              # YYYY-MM-DD
    is_past: bool = False
    is_valid: bool = True
    warnings: List[str] = []
    seasonality_alerts: List[str] = []
    error: Optional[str] = None


class ColumnMetadata(BaseModel):
    """Single column info from GET /experiments/get-columns-and-preview-data.
    Used to build column detector input."""
    name: str
    data_type: str                                  # "string", "integer", "float", "date", etc.
    cardinality: int
    null_pct: float = 0.0


class ColumnDetection(BaseModel):
    """Detection result for one column mapping."""
    column: Optional[str] = None
    columns: Optional[List[str]] = None             # for granularity (composite key)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: Optional[str] = None


class ColumnDetectorOutput(BaseModel):
    """GPT structured output: column metadata → column mappings."""
    granularity: ColumnDetection = ColumnDetection()
    date: ColumnDetection = ColumnDetection()
    date_frequency: Optional[str] = None            # "daily", "weekly", "monthly"
    kpi: ColumnDetection = ColumnDetection()
    group: ColumnDetection = ColumnDetection()      # measurement flow only
    features: List[str] = []
    blocking_factors: List[str] = []
    unmapped_columns: List[str] = []


class UpdateValidatorOutput(BaseModel):
    """GPT structured output: change request → update plan."""
    intent: UpdateIntent
    primary_field: Optional[str] = None
    update_payload: Dict[str, Any] = Field(default_factory=dict)
    cascade_updates: Dict[str, Any] = Field(default_factory=dict)
    requires_reprocessing: bool = False
    reprocessing_from_step: Optional[int] = None
    warnings: List[str] = []
    blocked_fields: List[str] = Field(default_factory=list)
    confirmation_needed: Optional[str] = None


# ────────────────────────────────────────────────────────────────
# BACKEND API PAYLOADS
# Exact shapes the backend expects. Agent builds these before
# calling POST/PUT endpoints.
# ────────────────────────────────────────────────────────────────


class ExperimentCreatePayload(BaseModel):
    """POST /v1/experiments
    Mirrors: ExperimentBaseSchema + ExperimentCreateSchema"""

    # ExperimentBaseSchema fields
    experiment_name: str = Field(..., min_length=1, max_length=255)
    problem_statement: Optional[str] = Field(None, max_length=1024)
    hypothesis: Optional[str] = Field(None, max_length=1024)
    smaller_sample_size: Optional[int] = None
    recommended_sample_size: Optional[int] = None
    metrices: Optional[List[Dict]] = None
    success_criteria: Optional[List[Dict]] = None
    group_configs: Optional[List[Dict]] = None
    interventions: Optional[List[Dict]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    baseline_start_date: Optional[date] = None
    baseline_end_date: Optional[date] = None
    step: int = 0
    is_draft: Optional[bool] = True
    is_rerun: Optional[bool] = False
    is_measurement_flow: Optional[bool] = None
    is_bu_prioritize_experiment: Optional[bool] = None
    exp_interventions: Optional[List[str]] = None

    # ExperimentCreateSchema fields
    experiment_id: Optional[str] = None
    experiment_type_id: Optional[str] = None
    country_id: Optional[str] = None
    use_case_id: Optional[str] = None
    business_function_id: Optional[str] = None
    hs_itrvn_occured: Optional[bool] = False


class ExperimentUpdatePayload(BaseModel):
    """PUT /v1/experiments/{literal}
    Mirrors: ExperimentBaseSchema + ExperimentUpdateSchema
    All optional — only send what changed."""

    # ExperimentBaseSchema fields
    experiment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    problem_statement: Optional[str] = Field(None, max_length=1024)
    hypothesis: Optional[str] = Field(None, max_length=1024)
    smaller_sample_size: Optional[int] = None
    recommended_sample_size: Optional[int] = None
    metrices: Optional[List[Dict]] = None
    success_criteria: Optional[List[Dict]] = None
    group_configs: Optional[List[Dict]] = None
    interventions: Optional[List[Dict]] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    baseline_start_date: Optional[date] = None
    baseline_end_date: Optional[date] = None
    step: Optional[int] = None
    is_draft: Optional[bool] = None
    is_rerun: Optional[bool] = None
    is_measurement_flow: Optional[bool] = None
    is_bu_prioritize_experiment: Optional[bool] = None
    exp_interventions: Optional[List[str]] = None

    # ExperimentUpdateSchema fields
    experiment_type_id: Optional[str] = None
    is_partial_measurement: Optional[bool] = None
    isDraft: Optional[bool] = None
    design_column_map: Optional[Dict] = None
    design_blocking_factors_overwrite: Optional[List[str]] = None
    design_table_filters: Optional[Dict] = None
    user_design_id: Optional[str] = None
    user_measurement_id: Optional[str] = None
    measurement_data: Optional[str] = None
    design_data_quality_report: Optional[Dict] = None
    total_group_size: Optional[int] = None
    country_id: Optional[str] = None
    use_case_id: Optional[str] = None
    business_function_id: Optional[str] = None
    file_name: Optional[str] = None
    design_table_name: Optional[str] = None
    user_created_table_name: Optional[str] = None
    filters: Optional[str] = None
    delta_master_table_name: Optional[str] = None
    delta_measurement_table_name: Optional[str] = None
    user_created_intervention_table_name: Optional[str] = None
    delta_intervention_table_name: Optional[str] = None
    measurement_column_map: Optional[Dict] = None
    measurement_table_filters: Optional[Dict] = None
    delta_recommendation_table_name: Optional[str] = None
    user_created_recommendation_table_name: Optional[str] = None
    recommendation_column_map: Optional[Dict] = None
    recommendation_table_filters: Optional[Dict] = None
    recommendation_data_quality_report: Optional[Dict] = None
    intervention_column_map: Optional[Dict] = None
    measurement_data_quality_report: Optional[Dict] = None
    measurement_group_size: Optional[int] = None
    status: Optional[int] = None
    design_config: Optional[Dict] = None
    measurement_config: Optional[dict] = None
    sql_design_table_name: Optional[str] = None
    sql_measurement_table_name: Optional[str] = None


class SampleSizePayload(BaseModel):
    """POST /v1/experiments/estimate-sample-size"""
    estimation_type: str
    mde: float = Field(..., gt=0)
    confidence_level: float = Field(..., gt=0)
    avg_baseline: float = Field(..., gt=0)
    std_deviation: Optional[float] = Field(None, gt=0)
    power_of_test: float = Field(..., gt=0)


class DataQualityPayload(BaseModel):
    """POST /v1/experiments/find-data-quality/{literal}
    Mirrors: DataQualityRequestSchema"""
    upload_type: Optional[str] = None               # "design" or "measurement"
    table_name: Optional[str] = None
    query: Optional[str] = None

# ────────────────────────────────────────────────────────────────
# CONVERSATION STATE
# Tracks the entire agent conversation across turns.
# ────────────────────────────────────────────────────────────────


class AgentMessage(BaseModel):
    """Single message in the conversation."""
    role: str                                       # "user", "agent", "system"
    content: str
    timestamp: Optional[datetime] = None
    message_type: MessageType = MessageType.TEXT


class CollectedFields(BaseModel):
    """All fields collected during conversation. Progressively filled."""

    # --- Flow ---
    flow_type: Optional[FlowType] = None
    is_measurement_flow: Optional[bool] = None

    # --- Hypothesis parser results ---
    hypothesis: Optional[str] = None
    primary_kpi: Optional[str] = None
    primary_kpi_literal: Optional[str] = None
    primary_kpi_type: Optional[str] = None
    primary_kpi_unit: Optional[str] = None
    expected_uplift: Optional[float] = None
    experiment_type_hint: Optional[ExperimentTypeHint] = None
    experiment_type_literal: Optional[str] = None
    control_action: Optional[str] = None
    treatment_action: Optional[str] = None
    secondary_kpis: Optional[List[str]] = None

    # --- Auto-generated ---
    experiment_name: Optional[str] = None
    problem_statement: Optional[str] = None

    # --- Auto-filled from profile ---
    country_id: Optional[str] = None
    zone_literal: Optional[str] = None
    business_function_id: Optional[str] = None
    currency_code: Optional[str] = None

    # --- Use case detection ---
    use_case_id: Optional[str] = None

    # --- Date validator results ---
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days: Optional[int] = None
    baseline_start: Optional[str] = None
    baseline_end: Optional[str] = None

    # --- Data upload ---
    file_name: Optional[str] = None
    upload_type: Optional[str] = None               # "design" or "measurement"
    total_rows: Optional[int] = None
    total_columns: Optional[int] = None
    data_quality_report: Optional[Dict] = None
    task_id: Optional[str] = None
    design_table_name: Optional[str] = None

    # --- Column detection results ---
    design_column_map: Optional[Dict] = None
    measurement_column_map: Optional[Dict] = None

    # --- Group configs ---
    group_configs: Optional[List[Dict]] = None
    total_group_size: Optional[int] = None
    control_allocation: float = 20.0
    treatment_allocation: float = 80.0

    # --- Sample size ---
    smaller_sample_size: Optional[int] = None
    recommended_sample_size: Optional[int] = None

    # --- Success criteria ---
    confidence_level: float = 95.0

    # --- After creation ---
    experiment_literal: Optional[str] = None


class ConversationState(BaseModel):
    """Full conversation state maintained across turns."""
    conversation_id: Optional[str] = None
    current_step: int = 0
    flow_type: Optional[FlowType] = None
    messages: List[AgentMessage] = []
    user_context: Optional[UserContext] = None
    collected: CollectedFields = CollectedFields()

    # Update flow
    update_experiment_literal: Optional[str] = None
    update_experiment_data: Optional[Dict] = None   # full experiment JSON
    experiments_list: Optional[List[ExperimentHistoryItem]] = None

    # Flags
    is_confirmed: bool = False
    is_complete: bool = False
    awaiting_upload: bool = False


# ────────────────────────────────────────────────────────────────
# AGENT API (frontend ↔ agent)
# ────────────────────────────────────────────────────────────────


class AgentChatRequest(BaseModel):
    """POST /api/v1/agent/chat — from frontend."""
    message: str
    conversation_id: Optional[str] = None
    auth_token: Optional[str] = None


class SuggestedOption(BaseModel):
    """Clickable option button in the chat UI."""
    label: str
    value: str


class AgentChatResponse(BaseModel):
    """Response from agent to frontend."""
    message: str
    conversation_id: str
    message_type: MessageType = MessageType.TEXT
    suggested_options: Optional[List[SuggestedOption]] = None
    current_step: int = 0
    flow_type: Optional[FlowType] = None
    requires_upload: bool = False
    metadata: Optional[Dict[str, Any]] = None