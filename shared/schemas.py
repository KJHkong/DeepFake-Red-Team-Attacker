"""Versioned boundary contracts; absent evidence stays null, never zero-filled.

Pydantic v2 patterns: https://docs.pydantic.dev/2.10/concepts/models/
"""
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

InputRole = Literal["source", "target", "reference", "image", "driving"]
ExecutionScope = Literal["full_pipeline", "generator_only", "content_only"]

def now() -> datetime:
    return datetime.now(timezone.utc)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Status(StrEnum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    INVALID_INPUT = "invalid_input"
    NO_FACE = "no_face"
    GENERATION_FAILED = "generation_failed"
    DETECTOR_FAILED = "detector_failed"
    EVALUATION_FAILED = "evaluation_failed"
    MODULE_FAILED = "module_failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNSUPPORTED_GENERATOR = "unsupported_generator"
    PLAN_FAILED = "plan_failed"


class Taxonomy(StrEnum):
    FACE_SWAPPING = "Deepfake/Face_Swapping"
    FACE_REENACTMENT = "Deepfake/Face_Reenactment"
    FACE_EDITING = "Deepfake/Face_Editing"
    GAN = "Synthetic/GAN_based"
    DIFFUSION = "Synthetic/Diffusion_based"
    AVATAR = "Synthetic/Avatar_Digital_Human"


class ErrorInfo(Contract):
    code: str
    message: str
    retryable: bool = False
    field: str | None = None


class FieldError(Contract):
    field: str
    code: str


class RequestError(ErrorInfo):
    run_id: str | None = None
    details: list[FieldError] = Field(default_factory=list)


class ErrorResponse(Contract):
    error: RequestError


class Budget(Contract):
    max_refines: int = Field(default=1, ge=0, le=1)
    max_duration_s: float = Field(default=60, gt=0, le=3600)


class Constraints(Contract):
    require_face: bool = True
    min_width: int = Field(default=64, ge=1)
    min_height: int = Field(default=64, ge=1)
    allow_business_display: bool = False
    required_capabilities: list[str] = Field(default_factory=list, max_length=16)
    preferred_strengths: list[str] = Field(default_factory=list, max_length=16)
    preferred_suitable_for: list[str] = Field(default_factory=list, max_length=16)


class InputAsset(Contract):
    role: InputRole
    path: str = Field(min_length=1, max_length=4096)
    display_allowed: bool = False


class InputPolicy(Contract):
    """Server-owned configuration, never part of a client AttackRequest."""
    allowed_roots: list[str] = Field(default_factory=list)
    max_file_bytes: int = Field(default=20 * 1024**2, ge=1, le=100 * 1024**2)
    max_total_bytes: int = Field(default=40 * 1024**2, ge=1, le=200 * 1024**2)
    max_pixels: int = Field(default=25_000_000, ge=1, le=100_000_000)
    allowed_formats: list[Literal["PNG", "JPEG", "WEBP"]] = Field(default_factory=lambda: ["PNG", "JPEG", "WEBP"], min_length=1)


class RoleRequirement(Contract):
    role: InputRole
    min_count: int = Field(default=0, ge=0, le=8)
    max_count: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def ordered_counts(self):
        if self.min_count > self.max_count:
            raise ValueError("min_count cannot exceed max_count")
        return self


class InputRequirements(Contract):
    roles: list[RoleRequirement] = Field(default_factory=list, max_length=5)
    prompt_required: bool = False
    accepts_prompt: bool = False

    @model_validator(mode="after")
    def unique_roles(self):
        if len({r.role for r in self.roles}) != len(self.roles):
            raise ValueError("Input requirement roles must be unique")
        if sum(r.min_count for r in self.roles) > 8:
            raise ValueError("Required input counts cannot exceed the request limit")
        return self


class ModifierOperation(Contract):
    tool_name: str = Field(min_length=1, max_length=128)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ModifierRequest(Contract):
    chain: list[ModifierOperation] | None = Field(default=None, max_length=16)
    preset: str | None = Field(default=None, min_length=1, max_length=128)
    enabled: bool = True
    tool_name: str | None = Field(default=None, min_length=1, max_length=128)
    required_capabilities: list[str] = Field(default_factory=list, max_length=16)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def disabled_is_empty(self):
        structured = self.chain is not None or self.preset is not None
        if self.chain is not None and self.preset is not None:
            raise ValueError("Modifier chain and preset are mutually exclusive")
        if structured and (self.tool_name or self.parameters or self.required_capabilities):
            raise ValueError("Chain/preset cannot be combined with legacy tool selection")
        if not self.enabled and (structured or self.tool_name or self.required_capabilities or self.parameters):
            raise ValueError("Disabled modifier cannot specify operations")
        return self


class AttackRequest(Contract):
    schema_version: Literal["0.1"] = "0.1"
    taxonomy: Taxonomy = Taxonomy.FACE_SWAPPING
    mode: Literal["mock", "validate_inputs", "real"] = "mock"
    execution_scope: ExecutionScope = "full_pipeline"
    generator: str | None = Field(default=None, min_length=1, max_length=128)
    modifier: ModifierRequest | None = None
    inputs: list[InputAsset] = Field(default_factory=list, max_length=8)
    prompt: str | None = Field(default=None, max_length=2000)
    seed: int = Field(default=42, ge=0, le=2**32-1)
    budget: Budget = Field(default_factory=Budget)
    constraints: Constraints = Field(default_factory=Constraints)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    mock_scenario: Literal["accepted", "gate1_reject", "gate2_reject", "refine_success",
                           "refine_reject", "generation_failed", "detector_failed", "no_face"] = "accepted"

    @model_validator(mode="after")
    def stage_scope_is_explicit(self):
        if self.execution_scope in {"generator_only", "content_only"} and self.mode != "real":
            raise ValueError("Stage execution scopes require real mode")
        if self.modifier and (self.modifier.chain is not None or self.modifier.preset is not None) and self.execution_scope != "content_only":
            raise ValueError("Structured Modifier requests require content_only")
        if self.execution_scope == "generator_only":
            if self.modifier is not None and self.modifier.enabled:
                raise ValueError("generator_only cannot execute a Modifier")
        return self


class ModifierPreset(Contract):
    operations: list[ModifierOperation] = Field(min_length=1, max_length=16)
    local_eval_status: Literal["not_run", "passed", "failed"] = "not_run"
    validated_recipe_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evidence_path: str | None = None


class ModifierConfig(Contract):
    max_operations: int = Field(default=8, ge=1, le=16)
    max_duration_s: float = Field(default=30, gt=0, le=120)
    max_pixels: int = Field(default=4_194_304, ge=1, le=25_000_000)
    max_file_bytes: int = Field(default=20 * 1024**2, ge=1, le=100 * 1024**2)
    presets: dict[str, ModifierPreset] = Field(default_factory=dict)


class ToolManifest(Contract):
    name: str
    version: str
    taxonomy: list[Taxonomy]
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    backend: str
    environment: str | None = None
    checkpoints: list[str] = Field(default_factory=list)
    controllable_parameters: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    mock: bool
    # ready/not_integrated remain readable for historical 0.1 payloads.
    availability: Literal["ready", "available", "not_integrated", "not_ready", "manifest_only", "disabled"] = "ready"
    input_requirements: InputRequirements = Field(default_factory=InputRequirements)
    kind: Literal["generator", "modifier"] = "generator"
    default_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    parameter_schema: dict[str, JsonValue] = Field(default_factory=dict)
    estimated_duration_s: float | None = Field(default=None, ge=0)
    output_width: int | None = Field(default=None, gt=0)
    output_height: int | None = Field(default=None, gt=0)
    display_name: str | None = None
    subtaxonomy: str | None = None
    task_tags: list[str] = Field(default_factory=list)
    output_type: Literal["image", "video"] = "image"
    worker_hint: str | None = None
    checkpoint_info: dict[str, JsonValue] = Field(default_factory=dict)
    default_priority: int | None = Field(default=None, ge=0)
    is_default: bool = False
    strengths: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    suitable_for: list[str] = Field(default_factory=list)
    expected_failure_modes: list[str] = Field(default_factory=list)
    runtime_notes: list[str] = Field(default_factory=list)
    resource_notes: list[str] = Field(default_factory=list)
    integration_risk: Literal["low", "medium", "high"] = "medium"
    local_eval_status: Literal["not_run", "historical_only", "passed", "failed"] = "not_run"
    ready_for_registry: bool = False
    mock_supported: bool = False
    planner_notes: list[str] = Field(default_factory=list)
    evidence: dict[str, JsonValue] = Field(default_factory=dict)
    randomness: Literal["deterministic", "stochastic"] | None = None
    seed_support: bool = False
    expected_effect: str | None = None

    @model_validator(mode="after")
    def defaults_are_supported(self):
        if set(self.default_parameters) - set(self.controllable_parameters):
            raise ValueError("Default parameters must be declared controllable parameters")
        return self


class Artifact(Contract):
    artifact_id: str = Field(default_factory=lambda: uuid4().hex)
    path: str
    media_type: str
    stage: str
    mock: bool
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    frame_count: int | None = Field(default=None, ge=1)
    sha256: str | None = None
    visibility: Literal["internal", "mock_demo"] = "internal"
    display_allowed: bool = False
    parent_ids: list[str] = Field(default_factory=list)
    role: InputRole | None = None
    source_path: str | None = None
    size_bytes: int | None = Field(default=None, gt=0)
    provenance: Literal["local_input_snapshot"] | None = None


class Candidate(Contract):
    candidate_id: str = Field(default_factory=lambda: uuid4().hex)
    artifact: Artifact
    revision: int = Field(default=0, ge=0, le=1)
    mock: bool
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PlannerConfig(Contract):
    generator_priority: list[str] = Field(default_factory=lambda: ["mock_image_generator", "face_swap", "text_to_image"])
    modifier_priority: list[str] = Field(default_factory=lambda: ["mock_image_modifier"])
    default_mock_modifier: str | None = "mock_image_modifier"
    overhead_reserve_s: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def priorities_are_unique(self):
        if any(len(items) != len(set(items)) for items in (self.generator_priority, self.modifier_priority)):
            raise ValueError("Priority lists must not contain duplicate names")
        return self


class PlanStep(Contract):
    step_id: str
    module: Literal["generator", "modifier", "reference_bank", "examiner", "refiner", "evaluator"]
    tool_name: str | None = None
    inputs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    output_ref: str
    condition: Literal["always", "gate1_failed", "refined", "gate1_passed"] = "always"
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ToolAssessment(Contract):
    tool_name: str
    kind: Literal["generator", "modifier"]
    priority: int | None = None
    eligible: bool
    issues: list[ErrorInfo] = Field(default_factory=list)
    version: str | None = None
    availability: str | None = None
    estimated_duration_s: float | None = Field(default=None, ge=0)
    preference_matches: list[str] = Field(default_factory=list)


class PlanRejection(Contract):
    generator: str
    modifier: str | None = None
    estimated_duration_s: float = Field(ge=0)
    error: ErrorInfo


class Plan(Contract):
    tool_name: str | None = None
    taxonomy: Taxonomy
    seed: int
    parameters: dict[str, JsonValue]
    mock: bool
    execution_scope: ExecutionScope = "full_pipeline"
    status: Literal["ready", "unsupported", "failed"] = "ready"
    strategy: str = Field(default="rule_based_v0.1", min_length=1, max_length=64)
    modifier_name: str | None = None
    modifier_chain: list[ModifierOperation] = Field(default_factory=list, max_length=16)
    modifier_recipe: dict[str, JsonValue] = Field(default_factory=dict)
    steps: list[PlanStep] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    input_dependencies: dict[str, list[str]] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    unmet_constraints: list[str] = Field(default_factory=list)
    deferred_constraints: list[str] = Field(default_factory=list)
    assessments: list[ToolAssessment] = Field(default_factory=list)
    rejected_combinations: list[PlanRejection] = Field(default_factory=list)
    estimated_duration_s: float | None = Field(default=None, ge=0)
    constraints: Constraints = Field(default_factory=Constraints)
    budget: Budget = Field(default_factory=Budget)
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def consistent_plan(self):
        if self.execution_scope != "content_only" and (self.modifier_chain or self.modifier_recipe):
            raise ValueError("Structured Modifier plans require content_only")
        if self.execution_scope == "content_only" and (self.mock or self.modifier_name is not None):
            raise ValueError("Content plans use real ordered operations, not legacy Modifier selection")
        if self.status != "ready" and (self.modifier_chain or self.modifier_recipe):
            raise ValueError("Failed plans cannot retain executable Modifier recipes")
        if self.execution_scope == "generator_only" and (self.mock or self.modifier_name is not None):
            raise ValueError("Generator-only plans cannot select Mock or Modifier execution")
        if self.status == "ready":
            if self.tool_name is None or self.error is not None:
                raise ValueError("Ready plans require a tool and no error")
        elif self.error is None or self.steps or self.tool_name is not None or self.modifier_name is not None:
            raise ValueError("Failed plans require an error and cannot contain executable selections")
        if self.steps:
            seen = set()
            examiner_seen = False
            for step in self.steps:
                if step.step_id in seen or not set(step.depends_on).issubset(seen):
                    raise ValueError("Plan steps need unique IDs and preceding dependencies")
                seen.add(step.step_id)
            if self.execution_order != [step.module for step in self.steps]:
                raise ValueError("Execution order must match plan steps")
            if sum(step.module == "generator" for step in self.steps) != 1:
                raise ValueError("Executable plans require exactly one Generator")
            if sum(step.module == "refiner" for step in self.steps) > self.budget.max_refines:
                raise ValueError("Plan exceeds refinement budget")
            expected = ["generator"] + (["modifier"] if self.modifier_name else []) + ["reference_bank", "examiner"]
            if any(step.module == "refiner" for step in self.steps):
                expected += ["refiner", "examiner"]
            expected += ["evaluator"]
            if self.execution_scope == "generator_only":
                expected = ["generator"]
            if self.execution_scope == "content_only":
                expected = ["generator"] + ["modifier"] * len(self.modifier_chain)
                for index, (step, operation) in enumerate(zip(self.steps[1:], self.modifier_chain), 1):
                    previous = self.steps[index-1]
                    if (step.tool_name != operation.tool_name or step.parameters != operation.parameters or
                            step.depends_on != [previous.step_id] or step.inputs != [previous.output_ref]):
                        raise ValueError("Content Modifier dependencies and parameters must match the ordered chain")
                if self.modifier_chain and not self.modifier_recipe.get("recipe_sha256"):
                    raise ValueError("Content Modifier plan requires a versioned recipe fingerprint")
            if self.execution_order != expected:
                raise ValueError("Plan must follow the supported pipeline and gate order")
            for step in self.steps:
                if step.module == "generator" and (step.tool_name != self.tool_name or step.parameters != self.parameters):
                    raise ValueError("Generator step must match the selected tool and parameters")
                if step.module == "modifier" and self.execution_scope != "content_only" and step.tool_name != self.modifier_name:
                    raise ValueError("Modifier step must match the selected modifier")
                expected_condition = "always"
                if step.module == "refiner":
                    expected_condition = "gate1_failed"
                elif step.module == "evaluator":
                    expected_condition = "gate1_passed"
                elif step.module == "examiner":
                    expected_condition = "refined" if examiner_seen else "always"
                    examiner_seen = True
                if step.condition != expected_condition:
                    raise ValueError("Plan conditions must preserve Gate 1 and one conditional refinement")
        return self


class ModuleResult(Contract):
    module: str
    status: Status
    mock: bool
    started_at: datetime
    elapsed_ms: float = Field(ge=0)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    data: dict[str, JsonValue] = Field(default_factory=dict)
    error: ErrorInfo | None = None
    candidate: Candidate | None = None


class ExaminerCheck(Contract):
    name: str = Field(min_length=1, max_length=128)
    status: Literal["passed", "failed", "measured", "not_evaluated", "unavailable", "error"] = "not_evaluated"
    required: bool = False
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    reason: str | None = None
    elapsed_ms: float = Field(default=0, ge=0)


class ExaminerResult(Contract):
    candidate_id: str
    validity_passed: bool | None
    distribution_check: Literal["passed", "failed", "not_evaluated"]
    constraints_passed: bool | None
    gate1_passed: bool | None = None
    face_present: bool | None = None
    reference_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float | None] = Field(default_factory=dict)
    mock: bool
    assessment_scope: Literal["legacy_pipeline", "per_image"] = "legacy_pipeline"
    execution_status: Status = Status.SUCCESS
    checks: list[ExaminerCheck] = Field(default_factory=list)
    artifact_id: str | None = None
    source_sha256: str | None = None
    image_info: dict[str, JsonValue] = Field(default_factory=dict)
    elapsed_ms: float = Field(default=0, ge=0)
    error: ErrorInfo | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def independent_assessment_is_not_a_gate(self):
        if self.assessment_scope == "per_image":
            if self.gate1_passed is not None or self.distribution_check != "not_evaluated":
                raise ValueError("Independent Examiner cannot declare a Gate or distribution decision")
            if self.execution_status != Status.SUCCESS and self.validity_passed is True:
                raise ValueError("Failed execution cannot certify image validity")
            if self.execution_status == Status.SUCCESS and any(c.status == "error" for c in self.checks):
                raise ValueError("Check execution errors must be reported as execution failures")
            required = [c for c in self.checks if c.required]
            if self.validity_passed is True and (not required or any(c.status != "passed" for c in required)):
                raise ValueError("Unknown, unexecuted or failed required checks cannot become valid")
            constraints = [c for c in self.checks if c.required and c.name in {"dimensions", "face_presence", "task_semantics"}]
            if self.constraints_passed is True and (not constraints or self.execution_status != Status.SUCCESS or
                    any(c.status != "passed" for c in constraints)):
                raise ValueError("Unverified task constraints cannot become passed")
        return self


class Gate1Decision(Contract):
    run_id: str
    candidate_id: str
    artifact_id: str
    source_sha256: str | None
    decision: Literal["passed", "rejected", "review", "error", "access_denied"]
    reasons: list[str] = Field(default_factory=list)
    rule_version: str = "per_image_gate1_day3_v1"
    # Distribution, retrieval and Detector scores are deliberately absent from this contract.


class ContentAssessment(Contract):
    evaluation_id: str
    run_id: str
    candidate_id: str
    artifact_id: str
    source_sha256: str
    examiner: ExaminerResult
    examiner_invocation: ModuleResult | None = None
    gate: Gate1Decision
    metadata_path: str
    attack_success: None = None
    gate2: None = None
    refiner_status: Literal["not_evaluated"] = "not_evaluated"

    @model_validator(mode="after")
    def consistent_assessment_binding(self):
        binding=(self.candidate_id,self.artifact_id,self.source_sha256)
        if (binding!=(self.examiner.candidate_id,self.examiner.artifact_id,self.examiner.source_sha256) or
                binding!=(self.gate.candidate_id,self.gate.artifact_id,self.gate.source_sha256) or
                self.run_id!=self.gate.run_id or self.examiner.assessment_scope!='per_image'):
            raise ValueError('Run, Candidate, Artifact and SHA bindings must agree across Examiner and Gate1')
        return self


class DetectorConfig(Contract):
    host: str = "127.0.0.1"
    port: int = Field(default=19004, ge=1, le=65535)
    timeout_ms: int = Field(default=30000, ge=1, le=120000)
    batch_size: int = Field(default=8, ge=1, le=8)
    max_image_bytes: int = Field(default=20 * 1024**2, ge=1)
    threshold: float = Field(default=0.5, ge=0, le=1)
    target: str = "External Deepfake Detector"
    configured_service_version: str = "unconfigured"
    configured_image: str | None = None
    deployment_mode: Literal["unverified", "python_source", "container", "remote"] = "unverified"
    backend_name: str | None = None
    declared_model_version: str | None = None
    version_evidence: dict[str, JsonValue] = Field(default_factory=dict)
    version_source: Literal["local_configuration_not_rpc"] = "local_configuration_not_rpc"
    production_threshold_confirmed: bool = False
    watermark_enabled_configured: bool | None = False
    watermark_threshold_configured: float | None = Field(default=None, ge=0, le=1)
    fusion_policy: str | None = None
    score_type: Literal["deepfake", "fused", "unknown"] | None = None

    @model_validator(mode="after")
    def determine_score_semantics(self):
        expected = "unknown" if self.watermark_enabled_configured is None else "fused" if self.watermark_enabled_configured else "deepfake"
        if self.score_type is None:
            self.score_type = expected
        elif expected != "unknown" and self.score_type != expected:
            raise ValueError("Score type contradicts the configured service watermark behavior")
        return self


class DetectorResult(Contract):
    artifact_id: str
    request_uuid: str
    status: Status
    mock: bool
    fake_score: float | None = Field(default=None, ge=0, le=1)
    real_score: float | None = Field(default=None, ge=0, le=1)
    deepfake_score: float | None = Field(default=None, ge=0, le=1)
    watermark_score: float | None = Field(default=None, ge=0, le=1)
    watermark_execution_status: str | None = None
    score_type: Literal["deepfake", "fused", "unknown"] | None = None
    service_status: int | None = None
    service_real_status: int | None = None
    service_cost_ms: int | None = Field(default=None, ge=0)
    rpc_elapsed_ms: float = Field(default=0, ge=0)
    crop_status: str | None = None
    service_level: int | None = None
    face_verified: Literal[False] = False
    predicted_fake: bool | None = None
    threshold: float = Field(ge=0, le=1)
    service: DetectorConfig
    backend_name: str | None = None
    declared_model_version: str | None = None
    version_evidence: dict[str, JsonValue] = Field(default_factory=dict)
    service_address: str | None = None
    predicted_label: Literal["fake", "real"] | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def require_evidence_for_scores(self):
        self.backend_name = self.service.backend_name
        self.declared_model_version = self.service.declared_model_version
        self.version_evidence = dict(self.service.version_evidence)
        self.service_address = f"{self.service.host}:{self.service.port}"
        if self.score_type is None:
            self.score_type = self.service.score_type
        if self.status == Status.SUCCESS:
            if self.fake_score is None or self.real_score is None or self.predicted_fake is None:
                raise ValueError("successful detector result requires scores and classification")
            if self.predicted_fake != (self.fake_score >= self.threshold):
                raise ValueError("classification must use configured threshold")
            label = "fake" if self.predicted_fake else "real"
            if self.predicted_label is not None and self.predicted_label != label:
                raise ValueError("predicted_label contradicts threshold classification")
            self.predicted_label = label
        elif any(x is not None for x in [self.fake_score, self.real_score, self.predicted_fake, self.predicted_label]):
            raise ValueError("failed execution cannot expose a usable detection score")
        return self


class HealthResult(Contract):
    status: Status
    elapsed_ms: float
    mock: bool
    error: ErrorInfo | None = None


class Gate2Decision(Contract):
    """Exploratory Detector classification, separate from legacy Mock Gate2."""
    run_id: str
    candidate_id: str
    artifact_id: str
    source_sha256: str | None
    gate1_decision: Literal['passed','rejected','review','error','access_denied']
    status: Literal['provisional_pass','provisional_fail','not_evaluated']
    detector_prediction: Literal['detector_predicted_real','detector_predicted_fake','not_evaluated'] = 'not_evaluated'
    fake_score: float | None = Field(default=None,ge=0,le=1)
    threshold: Literal[0.5] = 0.5
    threshold_kind: Literal['exploratory_0.5'] = 'exploratory_0.5'
    backend_name: str | None = None
    declared_model_version: str | None = None
    score_source: str = Field(min_length=1,max_length=1024)
    version_evidence: dict[str,JsonValue] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    attack_success: None = None

    @model_validator(mode='after')
    def exploratory_evidence_only(self):
        if self.status=='not_evaluated':
            if self.fake_score is not None or self.detector_prediction!='not_evaluated':
                raise ValueError('Unevaluated Gate2 cannot expose a usable prediction')
        else:
            if self.gate1_decision!='passed' or self.fake_score is None or not self.backend_name or not self.declared_model_version or not self.version_evidence:
                raise ValueError('Provisional Gate2 requires passed Gate1 and version-bound Detector evidence')
            expected='provisional_pass' if self.fake_score<self.threshold else 'provisional_fail'
            prediction='detector_predicted_real' if expected=='provisional_pass' else 'detector_predicted_fake'
            if self.status!=expected or self.detector_prediction!=prediction:
                raise ValueError('Gate2 classification contradicts exploratory threshold')
        return self


class RefinementResult(Contract):
    status: Literal['not_attempted','needs_review','unsupported','failed','refined']
    reason: str
    parent_candidate_id: str
    parent_artifact_id: str
    attempt_count: int = Field(default=0,ge=0,le=1)
    executed: bool = False
    derived_candidate: Candidate | None = None

    @model_validator(mode='after')
    def no_fictitious_refinement(self):
        if self.status=='refined':
            if not self.executed or self.attempt_count!=1 or self.derived_candidate is None or self.parent_artifact_id not in self.derived_candidate.artifact.parent_ids:
                raise ValueError('Refinement requires an actual derived Candidate linked to its parent Artifact')
            if self.derived_candidate.candidate_id==self.parent_candidate_id or self.derived_candidate.artifact.artifact_id==self.parent_artifact_id:
                raise ValueError('Refinement must retain independent derived identities')
        elif self.derived_candidate is not None:
            raise ValueError('Unexecuted/failed refinement cannot fabricate an output Candidate')
        if self.status=='not_attempted' and (self.executed or self.attempt_count):
            raise ValueError('Not attempted means zero executions')
        return self


class LLMConfig(Contract):
    base_url: Literal['https://api.deepseek.com'] = 'https://api.deepseek.com'
    model: str = 'deepseek-flash'
    timeout_seconds: float = Field(default=60,gt=0,le=120)
    max_retries: int = Field(default=1,ge=0,le=1)
    max_tokens: int = Field(default=4000,ge=256,le=8000)


class EvidenceFinding(Contract):
    statement: str = Field(min_length=1,max_length=1600)
    evidence: dict[str,JsonValue] = Field(min_length=1,max_length=16)


class ReportAnalysis(Contract):
    summary: str = Field(min_length=1,max_length=2400)
    findings: list[EvidenceFinding] = Field(min_length=1,max_length=12)
    limitations: list[str] = Field(max_length=12)
    human_review_suggestions: list[str] = Field(max_length=10)
    next_steps: list[str] = Field(max_length=10)
    uncertainties: list[str] = Field(max_length=12)


class PlanAdvice(Contract):
    task_summary: str = Field(min_length=1,max_length=1200)
    generator: str | None
    parameters: dict[str,JsonValue] = Field(default_factory=dict)
    modifier_chain: list[ModifierOperation] | None = None
    text_condition: str | None = Field(default=None,max_length=1000)
    constraints: list[str] = Field(default_factory=list,max_length=12)
    uncertainties: list[str] = Field(default_factory=list,max_length=12)


class LLMCallResult(Contract):
    llm_status: Literal['success','unavailable']
    task: Literal['report_analyst','plan_advisor']
    model_requested: str
    model_returned: str | None = None
    elapsed_ms: float = Field(default=0,ge=0)
    attempts: int = Field(default=0,ge=0,le=2)
    prompt_sha256: str | None = None
    system_prompt: str | None = None
    usage: dict[str,int] = Field(default_factory=dict)
    response_validated: bool = False
    data: dict[str,JsonValue] | None = None
    error: ErrorInfo | None = None


class CandidateSummary(Contract):
    schema_version: Literal['0.1'] = '0.1'
    run_id: str
    candidate_id: str
    artifact_id: str
    source_sha256: str
    generator: str
    modifier_chain: list[ModifierOperation]
    gate1: Gate1Decision
    gate2: Gate2Decision
    retrieval_summary: dict[str,JsonValue] = Field(default_factory=dict)
    refiner: RefinementResult
    attack_success: None = None
    evidence_refs: dict[str,str]

    @model_validator(mode='after')
    def linked_results_only(self):
        binding=(self.run_id,self.candidate_id,self.artifact_id,self.source_sha256)
        for gate in [self.gate1,self.gate2]:
            if binding!=(gate.run_id,gate.candidate_id,gate.artifact_id,gate.source_sha256):
                raise ValueError('CandidateSummary cannot combine unrelated gate evidence')
        if self.gate2.gate1_decision!=self.gate1.decision:
            raise ValueError('CandidateSummary cannot override Gate1 admission')
        if self.refiner.parent_candidate_id!=self.candidate_id or self.refiner.parent_artifact_id!=self.artifact_id:
            raise ValueError('Refiner assessment must refer to the same original Candidate')
        return self


class EvaluationResult(Contract):
    candidate_id: str
    status: Status
    detector: DetectorResult
    attack_success: bool | None
    gate2_passed: bool | None
    quality_metrics: dict[str, float | None] = Field(default_factory=dict)
    identity_metrics: dict[str, float | None] = Field(default_factory=dict)
    quality_evaluated: bool = False
    cost_ms: float = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)
    mock: bool


ReferenceLabel = Literal["real", "fake", "unknown"]
ReferenceRole = Literal["real_reference", "detected_fake", "missed_fake", "unknown"]
ReferenceUse = Literal["metadata_retrieval", "image_read", "display", "embedding", "evaluation"]
ReferenceSplit = Literal["retrieval_reference", "development", "holdout", "unknown"]
ReferenceID = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
ReferenceSHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ReferenceDetectorOutcome(Contract):
    predicted_label: ReferenceLabel = "unknown"
    outcome: Literal["detected", "missed", "unknown"] = "unknown"
    model_version: str | None = Field(default=None, min_length=1, max_length=128)
    source_reference: ReferenceID | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)


CaptureMethod = Literal["direct_digital", "suspected_screen_recapture", "confirmed_screen_recapture", "unknown"]
AttackScope = Literal["in_scope", "out_of_scope", "unknown"]
CaptureReviewStatus = Literal["unreviewed", "reviewed", "needs_review"]


class ReferenceQuery(Contract):
    candidate_id: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=3, ge=1, le=20)
    top_k: int | None = Field(default=None, ge=1, le=20)
    label: ReferenceLabel | None = None  # Historical filter alias, never a label inference rule.
    truth_label: ReferenceLabel | None = None
    directory_label: ReferenceLabel | None = None
    collection: ReferenceID | None = None
    capture_method: CaptureMethod | None = None
    attack_scope: AttackScope | None = None
    review_status: CaptureReviewStatus | None = None
    exclude_attack_scopes: list[AttackScope] = Field(default_factory=list, max_length=3)
    reference_role: ReferenceRole | None = None
    attack_family: str | None = Field(default=None, min_length=1, max_length=128)
    generation_method: str | None = Field(default=None, min_length=1, max_length=128)
    domain: str | None = Field(default=None, min_length=1, max_length=128)
    device: str | None = Field(default=None, min_length=1, max_length=128)
    scene: str | None = Field(default=None, min_length=1, max_length=128)
    split: ReferenceSplit = "retrieval_reference"
    min_width: int | None = Field(default=None, ge=1)
    max_width: int | None = Field(default=None, ge=1)
    min_height: int | None = Field(default=None, ge=1)
    max_height: int | None = Field(default=None, ge=1)
    require_display_allowed: bool = False
    required_permitted_use: list[ReferenceUse] = Field(default_factory=lambda: ["metadata_retrieval"], max_length=5)
    exclude_sample_ids: list[ReferenceID] = Field(default_factory=list, max_length=128)
    exclude_sha256: list[ReferenceSHA] = Field(default_factory=list, max_length=128)
    exclude_lineage_groups: list[ReferenceID] = Field(default_factory=list, max_length=128)
    manifest_version: ReferenceID | None = None

    @model_validator(mode="after")
    def consistent_aliases(self):
        if self.top_k is not None:
            if "limit" in self.model_fields_set and self.limit != self.top_k:
                raise ValueError("top_k and limit must agree")
            self.limit = self.top_k
        if self.label is not None:
            if self.truth_label is not None and self.truth_label != self.label:
                raise ValueError("label and truth_label filters must agree")
            self.truth_label = self.label
        for dimension in ("width", "height"):
            low, high = getattr(self, "min_"+dimension), getattr(self, "max_"+dimension)
            if low is not None and high is not None and low > high:
                raise ValueError("Minimum resolution cannot exceed maximum")
        return self


class ReferenceRecord(Contract):
    sample_id: ReferenceID
    artifact: Artifact
    label: str = "unknown"  # Retained historical annotation; does not populate truth_label.
    collection: str = Field(default="unknown", max_length=128)
    provenance: str = Field(default="unknown", max_length=4096)
    source_reference: ReferenceID | None = None
    lineage_group: ReferenceID | None = None
    truth_label: ReferenceLabel = "unknown"
    directory_label: ReferenceLabel = "unknown"  # Directory assignment, never verified truth.
    capture_method: CaptureMethod = "unknown"
    attack_scope: AttackScope = "unknown"
    review_status: CaptureReviewStatus = "unreviewed"
    review_source: ReferenceID | None = None
    annotation_version: ReferenceID | None = None
    truth_label_source: str | None = Field(default=None, min_length=1, max_length=256)
    reference_role: ReferenceRole = "unknown"
    detector_outcome: ReferenceDetectorOutcome | None = None
    attack_family: str | None = Field(default=None, min_length=1, max_length=128)
    generation_method: str | None = Field(default=None, min_length=1, max_length=128)
    domain: str | None = Field(default=None, min_length=1, max_length=128)
    device: str | None = Field(default=None, min_length=1, max_length=128)
    scene: str | None = Field(default=None, min_length=1, max_length=128)
    split: ReferenceSplit = "unknown"
    permission_status: Literal["authorized", "pending", "denied"] = "pending"
    permission_source: ReferenceID | None = None
    permitted_use: list[ReferenceUse] = Field(default_factory=list, max_length=5)
    validation_status: Literal["verified", "pending", "unknown"] = "unknown"
    verification_source: ReferenceID | None = None
    manifest_version: ReferenceID | None = None
    priority: int = Field(default=100, ge=0, le=10000)
    duplicate_sample_ids: list[ReferenceID] = Field(default_factory=list)
    rank: int | None = Field(default=None, ge=1)
    matched_conditions: dict[str, JsonValue] = Field(default_factory=dict)
    ranking_reason: str | None = None
    similarity_score: float | None = Field(default=None, ge=-1, le=1)
    similarity_metric: Literal["cosine"] | None = None

    @model_validator(mode="after")
    def evidence_is_explicit(self):
        if (self.capture_method != "unknown" or self.attack_scope != "unknown" or self.review_status == "reviewed") and not self.review_source:
            raise ValueError("Capture/scope review claims require explicit review provenance")
        if (self.capture_method in {"direct_digital", "confirmed_screen_recapture"} or self.attack_scope != "unknown") and self.review_status != "reviewed":
            raise ValueError("Confirmed capture or scope requires a completed explicit review")
        if self.permission_status == "authorized" and not self.permission_source:
            raise ValueError("Authorized references require a permission source")
        if self.validation_status == "verified":
            import re
            if (not self.verification_source or not self.manifest_version or not self.artifact.sha256 or
                    re.fullmatch(r"[a-f0-9]{64}", self.artifact.sha256) is None or
                    self.artifact.width is None or self.artifact.height is None or
                    self.artifact.media_type not in {"image/png", "image/jpeg", "image/webp"}):
                raise ValueError("Verified references require explicit image metadata and verification provenance")
            if self.truth_label != "unknown" and not self.truth_label_source:
                raise ValueError("Known truth requires an explicit label source")
        if ((self.reference_role == "real_reference" and self.truth_label == "fake") or
                (self.reference_role in {"detected_fake", "missed_fake"} and self.truth_label == "real")):
            raise ValueError("Reference role contradicts the known truth label")
        outcome = self.detector_outcome
        if outcome and outcome.outcome != "unknown":
            if self.truth_label != "fake" or not outcome.model_version or not outcome.source_reference:
                raise ValueError("Detected/missed outcomes require fake truth and Detector provenance")
            expected = "fake" if outcome.outcome == "detected" else "real"
            if outcome.predicted_label != expected:
                raise ValueError("Detector outcome contradicts the retained prediction")
        return self


class ReferenceManifest(Contract):
    schema_version: Literal["0.1"] = "0.1"
    manifest_version: ReferenceID
    record_count: int = Field(ge=0, le=20000)
    manifest_sha256: ReferenceSHA
    description: str = Field(default="", max_length=1024)


class ReferenceResult(Contract):
    query: ReferenceQuery
    status: Status = Status.SUCCESS
    references: list[ReferenceRecord] = Field(default_factory=list)
    manifest_version: str
    manifest_sha256: str
    matched_count: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(ge=0)
    image_pixels_read: bool = False
    retrieval_method: Literal["metadata_only", "visual_exact"] = "metadata_only"
    encoder: "EmbeddingSpec | None" = None
    index_version: ReferenceID | None = None
    query_sample_id: ReferenceID | None = None
    query_source_sha256: ReferenceSHA | None = None
    empty_reason: str | None = None
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def retrieval_evidence_is_consistent(self):
        if self.retrieval_method == "metadata_only" and (self.image_pixels_read or self.encoder is not None or
                self.index_version is not None or any(r.similarity_score is not None or r.similarity_metric is not None for r in self.references)):
            raise ValueError("Metadata-only retrieval cannot claim image or vector evidence")
        if self.retrieval_method == "visual_exact" and self.status == Status.SUCCESS:
            if self.encoder is None or self.index_version is None or self.query_source_sha256 is None:
                raise ValueError("Successful visual retrieval requires Encoder, Index and Query provenance")
            if any(r.similarity_score is None or r.similarity_metric != "cosine" for r in self.references):
                raise ValueError("Every visual hit requires a computed cosine score")
        return self

    def public_dict(self):
        """Explicit boundary projection. Internal Artifacts retain their confined paths."""
        result = self.model_dump(mode="json", exclude={"references":{"__all__":{
            "artifact":{"path", "source_path"}, "provenance":True, "permission_source":True}}})
        for record in result["references"]:
            source = record["truth_label_source"]
            if source and ("/" in source or "\\" in source):
                record["truth_label_source"] = "retained_private_source"
        return result


class EmbeddingSpec(Contract):
    """Public reproducibility metadata; never contains model paths or feature values."""
    encoder_name: ReferenceID
    version: ReferenceID
    weights_sha256: ReferenceSHA | None = None
    config_sha256: ReferenceSHA | None = None
    preprocessing_version: ReferenceID
    preprocessing: dict[str, JsonValue]
    feature_dimension: int = Field(ge=1, le=4096)
    normalization: Literal["l2"] = "l2"
    runtime: dict[str, str] = Field(default_factory=dict)
    implementation_sha256: ReferenceSHA | None = None


class VectorIndexMetadata(Contract):
    index_version: ReferenceID
    manifest_version: ReferenceID
    manifest_sha256: ReferenceSHA
    encoder: EmbeddingSpec
    record_count: int = Field(ge=0, le=128)
    entries_sha256: ReferenceSHA
    created_at: datetime = Field(default_factory=now)


ReferenceResult.model_rebuild()


class SheetPanel(Contract):
    label: str
    artifact: Artifact | None = None
    note: str = "N/A"
    text: list[str] = Field(default_factory=list)


class VerificationSheet(Contract):
    schema_version: Literal["0.1"] = "0.1"
    run_id: str
    mock: bool
    panels: list[SheetPanel]
    image_path: str
    spec_path: str
    metadata_path: str
    summary: list[str]


class RunMetadata(Contract):
    schema_version: Literal["0.1"] = "0.1"
    run_id: str
    state: Literal["running", "validated", "generated", "content_generated", "evaluated", "accepted", "rejected", "failed"] = "running"
    purpose: Literal["attack", "pipeline_smoke", "generator_stage", "content_generation"] = "attack"
    mock: bool
    request: AttackRequest
    started_at: datetime = Field(default_factory=now)
    finished_at: datetime | None = None
    plan: Plan | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    input_artifact_ids: list[str] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    final_candidate_id: str | None = None
    stages: list[ModuleResult] = Field(default_factory=list)
    examiner_history: list[ExaminerResult] = Field(default_factory=list)
    evaluation: EvaluationResult | None = None
    gate1: bool | None = None
    gate2: bool | None = None
    reference_ids: list[str] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=lambda: {"workflow": "0.1.0"})
    errors: list[ErrorInfo] = Field(default_factory=list)
    verification_sheet: VerificationSheet | None = None
    assessment_status: dict[str, Literal["not_evaluated"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def smoke_cannot_claim_attack_success(self):
        if self.request.execution_scope == "content_only" and self.purpose != "content_generation":
            raise ValueError("Content-only requests require content_generation provenance")
        if self.purpose == "content_generation":
            if (self.request.execution_scope != "content_only" or self.mock or self.gate1 is not None or
                    self.gate2 is not None or self.evaluation is not None or self.examiner_history or self.reference_ids or
                    self.state not in {"running", "content_generated", "failed"}):
                raise ValueError("Content generation cannot claim Gate or attack outcomes")
            if any(s.module not in {"planner", "input_ingestion", "generator", "modifier", "visualization"} or s.mock for s in self.stages):
                raise ValueError("Content generation cannot execute downstream or Mock modules")
            if any(c.mock or c.artifact.mock for c in self.candidates) or any(a.mock for a in self.artifacts):
                raise ValueError("Content generation requires real Artifacts")
            if self.state == "failed" and self.final_candidate_id is not None:
                raise ValueError("Failed content run has no final Candidate")
        if self.state == "content_generated":
            generator = [s for s in self.stages if s.module == "generator"]
            modifiers = [s for s in self.stages if s.module == "modifier"]
            if (self.purpose != "content_generation" or self.errors or not self.plan or self.plan.status != "ready" or
                    len(generator) != 1 or generator[0].status != Status.SUCCESS or not self.candidates or
                    self.final_candidate_id != self.candidates[-1].candidate_id or len(modifiers) != 1 or
                    modifiers[0].status != (Status.SUCCESS if self.plan.modifier_chain else Status.SKIPPED)):
                raise ValueError("Content state requires successful planned generation and complete Modifier execution")
            invocation = ModuleResult.model_validate(modifiers[0].data.get("modifier_invocation", {}))
            steps = invocation.data.get("step_results", [])
            if not isinstance(steps, list):
                raise ValueError("Content state requires structured Modifier results")
            previous = self.candidates[0].artifact.artifact_id
            artifact_ids = {a.artifact_id for a in self.artifacts}
            if len(steps) != len(self.plan.modifier_chain):
                raise ValueError("Content state requires every planned Modifier step")
            for result, operation in zip(steps, self.plan.modifier_chain):
                step_result = ModuleResult.model_validate(result)
                artifact = step_result.candidate.artifact if step_result.candidate else None
                if (artifact is None or step_result.status != Status.SUCCESS or step_result.error is not None or
                        step_result.inputs != [previous] or step_result.parameters != operation.parameters or
                        artifact.parent_ids != [previous] or artifact.artifact_id not in artifact_ids):
                    raise ValueError("Content state requires traceable successful Modifier results")
                previous = artifact.artifact_id
            if self.candidates[-1].artifact.artifact_id != previous:
                raise ValueError("Final Candidate must reference the last actual output")
        if self.request.execution_scope == "generator_only" and self.purpose != "generator_stage":
            raise ValueError("Generator-only requests require generator_stage provenance")
        if self.purpose == "generator_stage":
            if (self.request.execution_scope != "generator_only" or self.mock or self.gate1 is not None or
                    self.gate2 is not None or self.evaluation is not None or self.examiner_history or
                    self.reference_ids or self.state not in {"running", "generated", "failed"}):
                raise ValueError("Generator stage cannot claim a Gate or attack outcome")
            if any(s.module not in {"planner", "input_ingestion", "generator", "visualization"} or s.mock for s in self.stages):
                raise ValueError("Generator stage cannot execute unassessed downstream modules")
        if self.state == "generated" and (self.purpose != "generator_stage" or len(self.candidates) != 1 or
                self.candidates[0].mock or self.candidates[0].artifact.mock or self.errors or
                not any(s.module == "generator" and s.status == Status.SUCCESS for s in self.stages)):
            raise ValueError("Generated state requires one actual successful Generator candidate")
        if self.purpose == "pipeline_smoke":
            if self.mock or self.gate1 is not None or self.gate2 is not None or self.state in {"accepted","rejected"}:
                raise ValueError("Real pipeline smoke must leave formal Gates unassessed")
            if self.evaluation and (self.evaluation.attack_success is not None or self.evaluation.gate2_passed is not None):
                raise ValueError("Pipeline smoke cannot claim attack success")
        if self.state == "evaluated" and (self.purpose != "pipeline_smoke" or not self.evaluation or self.evaluation.status != Status.SUCCESS):
            raise ValueError("Evaluated state requires a successful Detector smoke result")
        return self
