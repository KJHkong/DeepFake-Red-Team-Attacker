"""Byte-bound observations and opt-in deterministic Source eligibility rules."""
from datetime import datetime, timezone
import hashlib
from time import perf_counter
from typing import Literal

from pydantic import Field, StrictInt, ValidationError
from shared.schemas import Contract, ErrorInfo, ModuleResult, Status


class Measurements(Contract):
    face_count: StrictInt | None = Field(default=None, ge=0, le=10000)
    yaw_degrees: float | None = Field(default=None, ge=-180, le=180)
    pitch_degrees: float | None = Field(default=None, ge=-180, le=180)
    roll_degrees: float | None = Field(default=None, ge=-180, le=180)
    eye_ratios: list[float] | None = Field(default=None, min_length=2, max_length=2)
    lip_ratio: float | None = Field(default=None, ge=0, le=2)
    confidence: float | None = Field(default=None, ge=0, le=1)
    quality: Literal['clear', 'ambiguous', 'unknown'] = 'unknown'
    reference_label: str | None = None
    measurement_semantics: str = 'unknown'
    raw_response_ref: str | None = None


class FilterConfig(Contract):
    enabled: bool = False
    max_abs_yaw_degrees: float = Field(default=8., gt=0, le=25)
    max_abs_pitch_degrees: float = Field(default=20., gt=0, le=45)
    closed_lip_ratio_max: float = Field(default=.03, ge=0, le=1)
    open_eye_ratio_min: float = Field(default=.20, ge=0, le=1)
    minimum_confidence_if_available: float = Field(default=.8, ge=0, le=1)
    require_reference_label: bool = True


class SourceStateAnalyzer:
    """Injected backend consumes validated bytes; never reopens an arbitrary path.

    A backend can be a real service or explicit SHA-bound historical measurements.
    The latter must be labelled as a temporary same-model proxy, not an NISP call.
    """
    def __init__(self, backend_call, *, backend, provenance=None):
        self.backend_call = backend_call
        self.backend = backend
        self.provenance = provenance or {}

    def analyze(self, artifact, data):
        started = perf_counter()
        result = ModuleResult(module='source_state', status=Status.SUCCESS, mock=False,
            started_at=datetime.now(timezone.utc), elapsed_ms=0, inputs=[artifact.artifact_id],
            data=dict(source_sha256=artifact.sha256, backend=self.backend,
                backend_provenance=self.provenance, eligibility='review', reasons=['not_filtered']))
        try:
            if (artifact.mock or not isinstance(data, bytes) or not 0 < len(data) <= 20*1024**2
                    or hashlib.sha256(data).hexdigest() != artifact.sha256):
                raise ValueError('source_input_changed')
            observation = Measurements.model_validate(self.backend_call(artifact, data))
            if observation.eye_ratios is not None and any(not 0 <= x <= 2 for x in observation.eye_ratios):
                raise ValueError('invalid_source_state')
            result.data['measurements'] = observation.model_dump(mode='json')
        except (TimeoutError,):
            result.status = Status.MODULE_FAILED
            result.error = ErrorInfo(code='source_state_timeout', message='Source backend timed out; state unknown', retryable=True)
        except (ValueError, ValidationError) as error:
            code = 'source_input_changed' if str(error) == 'source_input_changed' else 'invalid_source_state'
            result.status = Status.INVALID_INPUT
            result.error = ErrorInfo(code=code, message='Source bytes or returned measurements violate the contract')
        except Exception:
            result.status = Status.MODULE_FAILED
            result.error = ErrorInfo(code='source_state_backend_error', message='Source backend failed; state unknown')
        if result.error:
            result.data.update(measurements=None, reasons=[result.error.code])
        result.elapsed_ms = (perf_counter()-started)*1000
        return result


def filter_source(analysis, action, config=None):
    config = config or FilterConfig()
    result = analysis.model_copy(deep=True)
    result.parameters = dict(action=action, **config.model_dump())
    if not config.enabled:
        result.status = Status.SKIPPED
        result.data.update(eligibility='not_filtered', reasons=['source_filter_disabled'])
        return result
    if result.error:
        return result
    values = Measurements.model_validate(result.data['measurements'])
    decision, reasons = 'eligible', []
    def mark(state, reason):
        nonlocal decision
        # Known face-count mismatch is decisive; unknown is never converted to normal.
        if decision != 'ineligible':
            decision = state
        reasons.append(reason)
    if values.face_count is None:
        mark('review', 'face_count_unknown')
    elif values.face_count != 1:
        mark('ineligible', 'no_face' if values.face_count == 0 else 'multiple_faces')
    if values.quality != 'clear':
        mark('review', 'source_state_unclear')
    if values.confidence is not None and values.confidence < config.minimum_confidence_if_available:
        mark('review', 'low_confidence')
    required_label = {'mouth_closed_to_open': 'mouth_closed', 'frontal_to_left': 'head_frontal',
        'frontal_to_right': 'head_frontal', 'eyes_open_to_closed': 'eyes_open'}.get(action)
    if required_label is None:
        mark('review', 'unsupported_action')
    elif config.require_reference_label and values.reference_label != required_label:
        mark('review', 'reference_label_missing_or_mismatched')
    if action == 'mouth_closed_to_open':
        if values.lip_ratio is None:
            mark('review', 'lip_ratio_unknown')
        elif values.lip_ratio > config.closed_lip_ratio_max:
            mark('ineligible', 'mouth_not_closed')
    elif action in {'frontal_to_left', 'frontal_to_right'}:
        if values.yaw_degrees is None or values.pitch_degrees is None:
            mark('review', 'pose_unknown')
        elif (abs(values.yaw_degrees) > config.max_abs_yaw_degrees
                or abs(values.pitch_degrees) > config.max_abs_pitch_degrees):
            mark('ineligible', 'source_not_frontal')
    elif action == 'eyes_open_to_closed':
        if values.eye_ratios is None:
            mark('review', 'eye_ratios_unknown')
        elif min(values.eye_ratios) < config.open_eye_ratio_min:
            mark('ineligible', 'both_eyes_not_clearly_open')
    result.data.update(eligibility=decision, reasons=reasons or ['source_rules_satisfied'])
    return result


def execute_if_eligible(assessment, execute):
    """Optional pre-hook around the existing Orchestrator, not another executor."""
    if ((assessment.status == Status.SUCCESS and assessment.data['eligibility'] == 'eligible')
            or (assessment.status == Status.SKIPPED and assessment.parameters.get('enabled') is False)):
        return execute()
    return None
