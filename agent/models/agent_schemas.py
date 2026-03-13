"""Agent schemas — single source of truth for all data shapes.

Schema-driven approach: These Pydantic models are passed to OpenAI's
structured output (response_format) so GPT is FORCED to return valid
JSON matching these shapes. Prompts only explain WHY/HOW; schemas
define WHAT.

Every field here is cross-checked against the actual backend schemas in:
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
  - src/modules/currency/schemas/common.py
  - src/modules/volume/schemas/common.py
  - src/modules/roles/schemas/common.py
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════════
# SECTION 1 — ENUMS
# Mirrors: src/utils/enums.py
# ════════════════════════════════════════════════════════════════════


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


class EstimationType(str, Enum):
    """Mirrors: src/modules/experiments/schemas/sample_size.py"""

    CONTINUOUS = "continuous"
    PROPORTION = "proportion"


class UploadType(str, Enum):
    """Mirrors: src/utils/enums.UploadTypeEnum (subset used by agent)."""

    DESIGN = "design"
    MEASUREMENT = "measurement"


class TableType(str, Enum):
    """Table type for column preview / filter endpoints."""

    DESIGN = "Design"
    MEASUREMENT = "Measurement"


class ExperimentStatus(int, Enum):
    """Mirrors: src/utils/enums.ExperimentStatusTypes"""

    IN_PROGRESS = 100
    SUBMITTED = 101
    FAILED = 102
    POSITIVE_LIFT = 200
    NEGATIVE_LIFT = 300
    CANCELLED = 400
    INCONCLUSIVE = 401
    NO_IMPACT = 402


class ExperimentStep(int, Enum):
    """Mirrors: src/utils/enums.ExperimentStepTypes"""

    YET_TO_BEGIN = 0
    DATA_SELECTION = 1
    GROUP_EXPT_TYPE = 2
    GROUP_ALLOCATION = 3
    PENDING_SUBMISSION = 4
    MEASUREMENT_DATA_PENDING = 5
    SUBMITTED = 6


class MessageType(str, Enum):
    """Types of messages the agent can send to the frontend."""

    TEXT = "text"
    VALIDATION_DISPLAY = "validation_display"
    EXPERIMENT_DETAILS = "experiment_details"
    EXPERIMENT_LIST = "experiment_list"
    SUCCESS_SCREEN = "success_screen"
    OPTIONS = "options"
    DATA_PREVIEW = "data_preview"
    COLUMN_DETECTION = "column_detection"
    ERROR = "error"


# ════════════════════════════════════════════════════════════════════
# SECTION 2 — REFERENCE DATA SCHEMAS (read from backend)
# These mirror the backend GET responses. Agent reads these to
# validate user inputs and auto-fill fields.
# ════════════════════════════════════════════════════════════════════


class ZoneRef(BaseModel):
    """Mirrors: src/modules/zones/schemas/common.py → ZoneSchema"""

    id: int
    title: str
    description: Optional[str] = None
    zone_literal: Optional[str] = None


class CountryRef(BaseModel):
    """Mirrors: src/modules/countries/schemas/common.py → CountrySchema"""

    id: int
    title: str
    country_code: str
    currency_code: str
    country_literal: Optional[str] = None
    zone: Optional[ZoneRef] = None


class BusinessFunctionRef(BaseModel):
    """Mirrors: src/modules/business_functions/schemas/common.py → BusinessFunctionSchema"""

    id: int
    title: str
    business_function_literal: Optional[str] = None


class UseCaseRef(BaseModel):
    """Mirrors: src/modules/use_cases/schemas/common.py → UseCaseSchema"""

    title: str
    description: Optional[str] = None
    use_case_literal: Optional[str] = None
    business_functions: Optional[List[BusinessFunctionRef]] = None


class ExperimentTypeRef(BaseModel):
    """Mirrors: src/modules/experiment_types/schemas/common.py → ExperimentTypeSchema"""

    title: str
    description: Optional[str] = None
    experiment_type_literal: Optional[str] = None


class KpiRef(BaseModel):
    """Mirrors: src/modules/kpis/schemas/common.py → KPISchema"""

    name: str
    type: str  # "Continuous" or "Proportion"
    description: Optional[str] = None
    full_name: Optional[str] = None
    kpi_literal: Optional[str] = None


class CurrencyRef(BaseModel):
    """Mirrors: src/modules/currency/schemas/common.py → CurrencySchema"""

    id: int
    currency_code: str
    title: str
    is_default: bool
    exchange_rate: Optional[str] = None


class VolumeRef(BaseModel):
    """Mirrors: src/modules/volume/schemas/common.py → VolumeSchema"""

    id: int
    volume_code: str
    title: str
    is_default: bool


class RoleRef(BaseModel):
    """Mirrors: src/modules/roles/schemas/common.py → RoleSchema"""

    id: Optional[int] = None
    title: str
    description: Optional[str] = None
    role_literal: Optional[str] = None


class ReferenceData(BaseModel):
    """All master data fetched once at conversation start."""

    experiment_types: List[ExperimentTypeRef] = []
    kpis: List[KpiRef] = []
    countries: List[CountryRef] = []
    zones: List[ZoneRef] = []
    business_functions: List[BusinessFunctionRef] = []
    use_cases: List[UseCaseRef] = []
    currencies: List[CurrencyRef] = []
    volumes: List[VolumeRef] = []


# ════════════════════════════════════════════════════════════════════
# SECTION 3 — USER CONTEXT SCHEMAS
# Built from GET /auth/me + GET /users/{literal} +
# POST /experiments/get-all-experiments
# ════════════════════════════════════════════════════════════════════


class UserProfile(BaseModel):
    """Built from GET /v1/auth/me + GET /v1/users/{literal}.
    Mirrors: src/modules/users/schemas/common.py → UserSchema"""

    id: int
    user_literal: Optional[str] = None
    username: Optional[str] = None
    first_name: str
    last_name: Optional[str] = None
    email: str
    usage_reason: Optional[str] = None
    zones: Optional[List[ZoneRef]] = None
    countries: Optional[List[CountryRef]] = None
    business_functions: Optional[List[BusinessFunctionRef]] = None
    roles: Optional[List[RoleRef]] = None
    is_admin: Optional[bool] = None

    # Derived by agent from countries[0].currency_code
    primary_country_literal: Optional[str] = None
    primary_zone_literal: Optional[str] = None
    primary_business_function_literal: Optional[str] = None
    primary_currency_code: Optional[str] = None


class ExperimentHistoryItem(BaseModel):
    """Single experiment from user's history.
    Mirrors subset of: ExperimentListSchema"""

    experiment_literal: str
    experiment_name: Optional[str] = None
    status: Optional[Any] = None
    status_text: Optional[str] = None
    step: Optional[int] = None
    step_text: Optional[str] = None
    experiment_type_literal: Optional[str] = None
    experiment_type_title: Optional[str] = None
    use_case_literal: Optional[str] = None
    use_case_title: Optional[str] = None
    default_kpi: Optional[str] = None
    uplift: Optional[float] = None
    impact: Optional[float] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_measurement_flow: Optional[bool] = None
    is_draft: Optional[bool] = None
    created_at: Optional[datetime] = None


class HistoryStats(BaseModel):
    """Computed from user's experiment history for auto-fill."""

    most_used_use_case_literal: Optional[str] = None
    most_used_use_case_title: Optional[str] = None
    most_used_use_case_count: int = 0
    average_uplift_by_kpi: Optional[Dict[str, float]] = None  # kpi_literal → avg uplift
    average_duration_days: Optional[float] = None
    total_experiments: int = 0


class UserContext(BaseModel):
    """Complete user context loaded at conversation start."""

    profile: UserProfile
    experiments: List[ExperimentHistoryItem] = []
    history_stats: HistoryStats = HistoryStats()
    reference_data: ReferenceData = ReferenceData()


# ════════════════════════════════════════════════════════════════════
# SECTION 4 — SPECIALIST OUTPUT SCHEMAS
# These are passed to OpenAI's response_format so GPT is forced to
# return valid JSON matching these models. No output format needed
# in prompts.
# ════════════════════════════════════════════════════════════════════


# --- 4.1 Hypothesis Parser ---

class HypothesisParserInput(BaseModel):
    """Sent to the hypothesis parser specialist."""

    hypothesis_text: str
    actions_text: Optional[str] = None
    available_kpis: List[str] = []  # list of kpi names
    available_experiment_types: List[str] = []  # list of type titles


class HypothesisParserOutput(BaseModel):
    """Returned by GPT via structured output. Enforced by schema."""

    primary_kpi: Optional[str] = Field(None, description="Extracted KPI name or null")
    primary_kpi_match: Optional[str] = Field(None, description="Closest match from available_kpis")
    primary_kpi_literal: Optional[str] = Field(None, description="KPI literal from reference data")
    primary_kpi_type: Optional[str] = Field(None, description="continuous or proportion")
    expected_uplift: Optional[float] = Field(None, description="Expected percentage uplift or null")
    experiment_type_hint: ExperimentTypeHint = Field(
        default=ExperimentTypeHint.RCT, description="Detected experiment type"
    )
    control_action: Optional[str] = Field(None, description="What control group experiences")
    treatment_action: Optional[str] = Field(None, description="What treatment group experiences")
    target_segment: Optional[str] = Field(None, description="Target audience if mentioned")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Extraction confidence")


# --- 4.2 Date Validator ---

class DateValidatorInput(BaseModel):
    """Sent to the date validator specialist."""

    user_input: str
    current_date: str  # YYYY-MM-DD
    flow_type: FlowType


class DateValidatorOutput(BaseModel):
    """Returned by GPT via structured output. Enforced by schema."""

    start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    duration_days: Optional[int] = Field(None, description="Inclusive day count")
    baseline_start: Optional[str] = Field(None, description="YYYY-MM-DD, start_date minus 6 months")
    baseline_end: Optional[str] = Field(None, description="YYYY-MM-DD, start_date minus 1 day")
    baseline_duration_days: Optional[int] = None
    is_past: bool = False
    is_valid: bool = True
    warnings: List[str] = []
    seasonality_alerts: List[str] = []
    error: Optional[str] = None


# --- 4.3 Column Detector ---

class ColumnMetadata(BaseModel):
    """Single column info sent to the column detector."""

    name: str
    data_type: str  # "string", "integer", "float", "date", "datetime", "boolean"
    cardinality: int  # unique value count
    null_pct: float  # 0.0 to 1.0


class ColumnDetection(BaseModel):
    """Detection result for a single column mapping."""

    column: Optional[str] = None
    columns: Optional[List[str]] = None  # for granularity (can be composite)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: Optional[str] = None


class ColumnDetectorInput(BaseModel):
    """Sent to the column detector specialist."""

    columns: List[ColumnMetadata]
    masked_preview: Optional[List[Dict[str, str]]] = None  # masked sample rows
    total_rows: int
    user_kpi: str  # user's selected KPI name
    flow_type: FlowType


class ColumnDetectorOutput(BaseModel):
    """Returned by GPT via structured output. Enforced by schema."""

    granularity: ColumnDetection = ColumnDetection()
    date: ColumnDetection = ColumnDetection()
    date_frequency: Optional[str] = Field(None, description="daily, weekly, or monthly")
    kpi: ColumnDetection = ColumnDetection()
    group: ColumnDetection = ColumnDetection()  # measurement flow only
    features: List[str] = []
    blocking_factors: List[str] = []
    unmapped_columns: List[str] = []


# --- 4.4 Update Validator ---

class UpdateIntent(str, Enum):
    """What the update validator determined."""

    UPDATE_FIELD = "update_field"
    RE_UPLOAD_DATA = "re_upload_data"
    SWITCH_TYPE = "switch_type"
    CANNOT_UPDATE = "cannot_update"


class UpdateValidatorInput(BaseModel):
    """Sent to the update validator specialist."""

    current_experiment: Dict[str, Any]  # full experiment object from GET /experiments/{literal}
    user_request: str  # raw change request text
    experiment_status: Optional[str] = None  # "draft", "active", "completed", "paused"
    current_date: str  # YYYY-MM-DD


class UpdateValidatorOutput(BaseModel):
    """Returned by GPT via structured output. Enforced by schema."""

    intent: UpdateIntent
    primary_field: Optional[str] = Field(None, description="Main field being changed")
    update_payload: Dict[str, Any] = Field(default_factory=dict, description="Fields to directly update")
    cascade_updates: Dict[str, Any] = Field(default_factory=dict, description="Auto-calculated dependent fields")
    requires_reprocessing: bool = False
    reprocessing_from_step: Optional[int] = Field(None, description="Step number to re-run from, 1-6")
    warnings: List[str] = []
    blocked_fields: List[str] = Field(default_factory=list, description="Fields that cannot be updated with reason")
    confirmation_needed: Optional[str] = Field(None, description="Message to show before applying")


# ════════════════════════════════════════════════════════════════════
# SECTION 5 — BACKEND API PAYLOADS
# These mirror the EXACT shapes the backend expects. Agent builds
# these before calling POST/PUT endpoints.
# ════════════════════════════════════════════════════════════════════


# --- 5.1 Sub-schemas used inside experiment payloads ---

class SuccessCriteriaPayload(BaseModel):
    """Mirrors: src/modules/experiments/schemas/common.py → SuccessCriteria"""

    kpi_literal: Optional[str] = None
    minimum_uplift: Optional[float] = None
    confidence_level: Optional[float] = None


class MetricPayload(BaseModel):
    """Structure inside metrices[] array.
    Mirrors: src/modules/experiments/schemas/experience_metadatas.py → Metric"""

    kpi_literal: Optional[str] = None
    literral_value: Optional[str] = None  # display name (note: typo matches backend)
    kpi_type: Optional[str] = None  # "continuous" or "proportion"


class GroupConfigPayload(BaseModel):
    """Structure inside group_configs[] array.
    Mirrors: src/modules/experiments/schemas/experience_metadatas.py → Group"""

    name: str
    action: str
    is_control: bool
    group_allocation_percentage: Optional[float] = None
    group_size: Optional[int] = None


class InterventionPayload(BaseModel):
    """Structure inside interventions[] array (MTE only).
    Mirrors: src/modules/experiments/schemas/experience_metadatas.py → Intervention"""

    intervention_name: str
    group_configs: List[GroupConfigPayload] = []


class DesignColumnMapPayload(BaseModel):
    """Mirrors: src/modules/experiments/schemas/common.py → DesignColumnMapSchema"""

    granularities: Optional[List[str]] = None
    blocking_factors: Optional[List[str]] = None
    kpi: Optional[Union[str, List[dict]]] = None
    features: Optional[List[str]] = None
    group_column: Optional[str] = None
    allocation_group: Optional[str] = None
    date: Optional[str] = None
    date_frequency: Optional[str] = None


class DesignTableFilterPayload(BaseModel):
    """Mirrors: src/modules/experiments/schemas/experience_metadatas.py → DesignTableFilterSchema"""

    filters: Dict[str, List[Any]] = {}
    sql_query: Optional[str] = None


class AutotuneConfigPayload(BaseModel):
    """Mirrors: src/modules/experiments/schemas/common.py → AutotuneConfigSchema"""

    is_autotune_enabled: bool = False
    n_match: List[int] = Field(default_factory=lambda: [1])
    caliper: List[float] = Field(default_factory=list)


# --- 5.2 Experiment Create Payload ---

class ExperimentCreatePayload(BaseModel):
    """Mirrors: src/modules/experiments/schemas/common.py → ExperimentCreateSchema
    (ExperimentBaseSchema + ExperimentCreateSchema fields)

    This is the EXACT payload sent to POST /v1/experiments.
    """

    # --- From ExperimentBaseSchema ---
    experiment_name: str = Field(..., min_length=1, max_length=255)
    problem_statement: Optional[str] = Field(None, max_length=1024)
    hypothesis: Optional[str] = Field(None, max_length=1024)
    smaller_sample_size: Optional[int] = None
    recommended_sample_size: Optional[int] = None
    metrices: Optional[List[Dict]] = None  # List[MetricPayload] serialized
    success_criteria: Optional[List[SuccessCriteriaPayload]] = None
    group_configs: Optional[List[Dict]] = None  # List[GroupConfigPayload] serialized
    interventions: Optional[List[Dict]] = None  # List[InterventionPayload] serialized (MTE)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    baseline_start_date: Optional[date] = None
    baseline_end_date: Optional[date] = None
    step: int = 0
    is_draft: Optional[bool] = True
    is_rerun: Optional[bool] = False
    is_measurement_flow: Optional[bool] = None
    is_bu_prioritize_experiment: Optional[bool] = None
    exp_interventions: Optional[List[str]] = None  # MTE intervention names

    # --- From ExperimentCreateSchema ---
    experiment_id: Optional[str] = None
    experiment_type_id: Optional[str] = None  # experiment_type_literal
    country_id: Optional[str] = None  # country_literal
    use_case_id: Optional[str] = None  # use_case_literal
    business_function_id: Optional[str] = None  # business_function_literal
    hs_itrvn_occured: Optional[bool] = False  # True for OS measurement flow


# --- 5.3 Experiment Update Payload ---

class ExperimentUpdatePayload(BaseModel):
    """Mirrors: src/modules/experiments/schemas/common.py → ExperimentUpdateSchema
    (ExperimentBaseSchema + ExperimentUpdateSchema fields)

    This is the EXACT payload sent to PUT /v1/experiments/{literal}.
    All fields optional — only send what changed.
    """

    # --- From ExperimentBaseSchema ---
    experiment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    problem_statement: Optional[str] = Field(None, max_length=1024)
    hypothesis: Optional[str] = Field(None, max_length=1024)
    smaller_sample_size: Optional[int] = None
    recommended_sample_size: Optional[int] = None
    metrices: Optional[List[Dict]] = None
    success_criteria: Optional[List[SuccessCriteriaPayload]] = None
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

    # --- From ExperimentUpdateSchema ---
    experiment_type_id: Optional[str] = None
    is_partial_measurement: Optional[bool] = None
    isDraft: Optional[bool] = None  # note: backend has BOTH is_draft and isDraft
    design_column_map: Optional[DesignColumnMapPayload] = None
    design_blocking_factors_overwrite: Optional[List[str]] = None
    design_table_filters: Optional[DesignTableFilterPayload] = None
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
    filters: Optional[str] = None  # JSON string
    delta_master_table_name: Optional[str] = None
    delta_measurement_table_name: Optional[str] = None
    user_created_intervention_table_name: Optional[str] = None
    delta_intervention_table_name: Optional[str] = None
    measurement_column_map: Optional[Dict] = None
    measurement_table_filters: Optional[DesignTableFilterPayload] = None
    delta_recommendation_table_name: Optional[str] = None
    user_created_recommendation_table_name: Optional[str] = None
    recommendation_column_map: Optional[Dict] = None
    recommendation_table_filters: Optional[DesignTableFilterPayload] = None
    recommendation_data_quality_report: Optional[Dict] = None
    intervention_column_map: Optional[Dict] = None
    measurement_data_quality_report: Optional[Dict] = None
    measurement_group_size: Optional[int] = None
    status: Optional[int] = None
    design_config: Optional[AutotuneConfigPayload] = None
    measurement_config: Optional[dict] = None
    sql_design_table_name: Optional[str] = None
    sql_measurement_table_name: Optional[str] = None


# --- 5.4 Sample Size Estimation ---

class SampleSizePayload(BaseModel):
    """Mirrors: src/modules/experiments/schemas/sample_size.py → SampleSizeRequest"""

    estimation_type: EstimationType
    mde: float = Field(..., gt=0)
    confidence_level: float = Field(..., gt=0)
    avg_baseline: float = Field(..., gt=0)
    std_deviation: Optional[float] = Field(None, gt=0)
    power_of_test: float = Field(..., gt=0)


class SampleSizeResult(BaseModel):
    """Response from POST /v1/experiments/estimate-sample-size."""

    min_test_sample_size: int
    min_control_sample_size: int
    assumption: Optional[str] = None


# --- 5.5 Experiment List Filter ---

class ExperimentListFilter(BaseModel):
    """Subset of ExperimentListFilterSchema used by agent.
    Mirrors: src/modules/experiments/schemas/common.py → ExperimentListFilterSchema"""

    only_my_experiments: Optional[bool] = None
    created_by_ids: Optional[str] = None
    is_draft: Optional[bool] = None
    statuses: Optional[str] = None
    experiment_type: Optional[str] = None
    search: Optional[str] = None
    country_ids: Optional[str] = None
    use_case_ids: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# --- 5.6 Data Preview Filter ---

class DataPreviewFilter(BaseModel):
    """Mirrors: src/modules/experiments/schemas/experience_metadatas.py → DesignDataPreviewFilterSchema"""

    limit: int = 10
    filters: Dict[str, List[Any]] = {}


# ════════════════════════════════════════════════════════════════════
# SECTION 6 — BACKEND RESPONSE SCHEMAS
# Shapes the agent receives back from GET endpoints. Used to parse
# and display data to the user.
# ════════════════════════════════════════════════════════════════════


class ExperimentMetadataResponse(BaseModel):
    """Mirrors: src/modules/experiments/schemas/experience_metadatas.py → ExperimentMetadataSchema"""

    experiment_id: Optional[int] = None
    metrices: Optional[List[Dict]] = None
    group_configs: Optional[List[Dict]] = None
    interventions: Optional[List[Dict]] = None
    design_column_map: Optional[Dict] = None
    design_table_filters: Optional[DesignTableFilterPayload] = None
    measurement_column_map: Optional[Dict] = None
    intervention_column_map: Optional[Dict] = None
    design_data_quality_report: Optional[Dict] = None
    measurement_data_quality_report: Optional[Dict] = None
    additional_design_data_quality_report: Optional[Dict] = None
    additional_measurement_data_quality_report: Optional[Dict] = None
    recommendation_column_map: Optional[Dict] = None
    recommendation_data_quality_report: Optional[Dict] = None
    recommendation_table_filters: Optional[DesignTableFilterPayload] = None
    measurement_group_size: Optional[int] = None
    measurement_table_filters: Optional[DesignTableFilterPayload] = None
    intervention_data_quality_report: Optional[Dict] = None
    total_group_size: Optional[int] = None
    self_service_insight_filter: Optional[List[Dict]] = None
    design_config: Optional[Dict] = None
    measurement_config: Optional[Dict] = None


class ExperimentDetailResponse(BaseModel):
    """Full experiment response from GET /v1/experiments/{literal}.
    Mirrors: src/modules/experiments/schemas/common.py → ExperimentSchema

    Includes fields from ExperimentBaseSchema + ExperimentCommonSchema +
    ExperimentListSchema + ExperimentSchema."""

    # --- Identity ---
    id: Optional[int] = None
    experiment_literal: Optional[str] = None

    # --- Core fields (ExperimentBaseSchema) ---
    experiment_name: Optional[str] = None
    problem_statement: Optional[str] = None
    hypothesis: Optional[str] = None
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

    # --- ExperimentCommonSchema fields ---
    uplift: Optional[float] = None
    insights: Optional[str] = None
    impact: Optional[float] = None
    parent_experiment_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    experiment_metadatas: Optional[ExperimentMetadataResponse] = None
    success_criteria: Optional[List[Dict]] = None
    status: Optional[Any] = None
    data_mappings: Optional[Dict] = None
    files: Optional[List[Dict]] = None
    is_sdk: Optional[int] = None
    exp_type_status: Optional[str] = None
    actual_lift_imcrement_on_success_criteria: Optional[float] = None
    is_top_learnings: Optional[bool] = None
    is_recurring_child_experiment: Optional[bool] = None

    # --- ExperimentListSchema fields ---
    experiment_type: Optional[Dict] = None  # {title, experiment_type_literal}
    country: Optional[Dict] = None  # {title, country_code, currency_code, country_literal}
    use_case: Optional[Dict] = None  # {title, use_case_literal}
    business_function: Optional[Dict] = None  # {title, business_function_literal}
    user: Optional[Dict] = None  # {first_name, last_name, email, user_literal}
    status_text: Optional[str] = None
    failure_short_message: Optional[str] = None
    step_text: Optional[str] = None
    default_kpi: Optional[str] = None
    is_partial: Optional[bool] = None
    is_measurement_triggered: Optional[bool] = None
    is_completed: Optional[bool] = None
    is_partial_measurement: Optional[bool] = None
    measurement_confidence_level: Optional[float] = None
    is_scheduled: Optional[bool] = None
    blocking_factors_formated: Optional[List[str]] = None

    # --- ExperimentSchema fields ---
    parent_experiment: Optional[Dict] = None


class ColumnPreviewResponse(BaseModel):
    """Response from GET /v1/experiments/get-columns-and-preview-data/{literal}."""

    columns: List[ColumnMetadata] = []
    preview_data: Optional[List[Dict[str, Any]]] = None
    total_rows: Optional[int] = None


class TaskStatusResponse(BaseModel):
    """Response from GET /v1/experiments/task/status/{task_id}."""

    task_id: str
    status: str  # "PENDING", "STARTED", "SUCCESS", "FAILURE"
    result: Optional[Any] = None


class DataQualityResponse(BaseModel):
    """Response from POST /v1/experiments/find-data-quality/{literal}."""

    total_rows: Optional[int] = None
    total_columns: Optional[int] = None
    missing_values: Optional[Dict[str, Any]] = None
    column_types: Optional[Dict[str, str]] = None
    quality_score: Optional[float] = None
    report: Optional[Dict] = None


# ════════════════════════════════════════════════════════════════════
# SECTION 7 — CONVERSATION STATE
# Tracks the entire agent conversation across turns.
# ════════════════════════════════════════════════════════════════════


class AgentMessage(BaseModel):
    """Single message in the conversation."""

    role: str  # "user", "agent", "system"
    content: str
    timestamp: Optional[datetime] = None
    message_type: MessageType = MessageType.TEXT
    metadata: Optional[Dict[str, Any]] = None  # extra data (options, display fields)


class CollectedFields(BaseModel):
    """All fields collected during the conversation so far.
    Progressively filled as the user provides information."""

    # --- Flow ---
    flow_type: Optional[FlowType] = None
    is_measurement_flow: Optional[bool] = None

    # --- From hypothesis parser ---
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
    target_segment: Optional[str] = None
    secondary_kpis: Optional[List[str]] = None

    # --- Auto-generated ---
    experiment_name: Optional[str] = None
    problem_statement: Optional[str] = None

    # --- From user profile (auto-filled) ---
    country_id: Optional[str] = None
    zone_literal: Optional[str] = None
    business_function_id: Optional[str] = None
    currency_code: Optional[str] = None

    # --- From use case detection ---
    use_case_id: Optional[str] = None
    use_case_title: Optional[str] = None

    # --- From date validator ---
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days: Optional[int] = None
    baseline_start: Optional[str] = None
    baseline_end: Optional[str] = None

    # --- From data upload ---
    file_name: Optional[str] = None
    upload_type: Optional[UploadType] = None
    data_quality_report: Optional[Dict] = None
    total_rows: Optional[int] = None
    total_columns: Optional[int] = None
    task_id: Optional[str] = None

    # --- From column detector ---
    design_column_map: Optional[DesignColumnMapPayload] = None
    measurement_column_map: Optional[Dict] = None
    design_table_filters: Optional[DesignTableFilterPayload] = None
    measurement_table_filters: Optional[DesignTableFilterPayload] = None

    # --- Group configs ---
    group_configs: Optional[List[GroupConfigPayload]] = None
    interventions: Optional[List[InterventionPayload]] = None
    exp_interventions: Optional[List[str]] = None
    total_group_size: Optional[int] = None

    # --- Sample size ---
    sample_size_result: Optional[SampleSizeResult] = None

    # --- Success criteria ---
    success_criteria: Optional[List[SuccessCriteriaPayload]] = None
    confidence_level: float = 95.0
    control_allocation: float = 20.0
    treatment_allocation: float = 80.0

    # --- Experiment result (after creation) ---
    experiment_literal: Optional[str] = None

    # --- Design config ---
    design_config: Optional[AutotuneConfigPayload] = None
    measurement_config: Optional[dict] = None


class ConversationState(BaseModel):
    """Full conversation state maintained across turns."""

    conversation_id: Optional[str] = None
    current_step: int = 0  # maps to conversation flow step (1-7 create, UPDATE STEP 1-6)
    flow_type: Optional[FlowType] = None
    messages: List[AgentMessage] = []
    user_context: Optional[UserContext] = None
    collected: CollectedFields = CollectedFields()

    # --- Update flow state ---
    update_experiment_literal: Optional[str] = None  # which experiment is being updated
    update_experiment_data: Optional[ExperimentDetailResponse] = None  # full experiment for display
    experiments_list: Optional[List[ExperimentHistoryItem]] = None  # user's experiments for selection

    # --- Flags ---
    is_confirmed: bool = False  # user confirmed validation display
    is_complete: bool = False  # conversation finished
    awaiting_upload: bool = False  # waiting for file upload
    specialist_retries: int = 0  # retry counter for specialist failures


# ════════════════════════════════════════════════════════════════════
# SECTION 8 — AGENT API REQUEST / RESPONSE
# What the frontend sends and receives.
# ════════════════════════════════════════════════════════════════════


class AgentChatRequest(BaseModel):
    """Request from frontend to POST /api/v1/agent/chat."""

    message: str
    conversation_id: Optional[str] = None
    auth_token: Optional[str] = None  # Bearer token for backend API calls


class SuggestedOption(BaseModel):
    """A clickable option button shown in the chat."""

    label: str  # display text
    value: str  # value sent back when clicked


class AgentChatResponse(BaseModel):
    """Response from agent back to the frontend."""

    message: str
    conversation_id: str
    message_type: MessageType = MessageType.TEXT
    suggested_options: Optional[List[SuggestedOption]] = None
    current_step: int = 0
    flow_type: Optional[FlowType] = None
    requires_upload: bool = False  # frontend should show file upload UI
    metadata: Optional[Dict[str, Any]] = None  # extra data for rendering (e.g. experiment details)


class AgentUploadRequest(BaseModel):
    """Metadata sent alongside file upload to POST /api/v1/agent/upload."""

    conversation_id: str
    upload_type: UploadType = UploadType.DESIGN
    auth_token: Optional[str] = None