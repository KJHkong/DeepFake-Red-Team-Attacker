"""Five-action source rules, explicitly separate from historical filter defaults."""
from copy import deepcopy
import math
from pydantic import Field
from shared.schemas import Contract, ModuleResult, Status
from modules.source_state.analyzer import Measurements

SUPPORTED = ('mouth_closed_to_open', 'mouth_open_to_closed', 'eyes_open_to_closed', 'frontal_to_left', 'frontal_to_right')


class SourceRules(Contract):
    version: str = 'source_five_actions_v1_1028'
    enabled: bool = False
    calibration: dict[str, dict] = Field(default_factory=lambda: {
        group: dict(status='pending_manual_review', evidence=[]) for group in ['mouth', 'eyes', 'frontal']})


def select_profile(action, level=None):
    if action not in SUPPORTED:
        raise ValueError('Unsupported conversion; no implicit alternate action')
    if action.startswith('mouth_'):
        from modules.generator.adapters.mouth_actions import load_bank, VERSION, BANK_SHA
        rows = load_bank()['entries']
    else:
        from modules.generator.adapters.day9_actions import entries, VERSION, BANK_SHA
        rows = entries()
    if level is None:
        level = 'canonical' if action in ('mouth_open_to_closed', 'eyes_open_to_closed') else 'L1'
    match = next((e for e in rows if e['action'] == action and e['intensity_level'] == level), None)
    if match is None:
        raise ValueError('Unknown action intensity')
    if action.startswith('mouth_'):
        frozen_version, frozen_sha = VERSION, BANK_SHA
    else:
        from modules.generator.adapters.frozen_actions import load_bank, VERSION as FROZEN_VERSION, BANK_SHA as FROZEN_SHA
        if match not in load_bank()['entries']:raise ValueError('Worker mapping differs from frozen profile')
        frozen_version, frozen_sha = FROZEN_VERSION, FROZEN_SHA
    return dict(deepcopy(match), worker_bank_version=VERSION, bank_sha256=BANK_SHA,
                frozen_profile_version=frozen_version, frozen_config_sha256=frozen_sha,
                parameters=dict(prompt_id=match['prompt_id'], bank_version=VERSION),
                native_text_conditioning=False)


def assess_source(analysis, action, *, level=None, config=None, manual_label='unknown', manual_trusted=False):
    config = config or SourceRules()
    result = analysis.model_copy(deep=True)
    result.module = 'source_rule_filter'
    try:
        profile = select_profile(action, level)
    except ValueError:
        result.data.update(eligibility='review', reasons=['unsupported_action_or_level'], selected_profile=None)
        return result
    result.parameters = dict(action=action, rule_version=config.version, enabled=config.enabled, manual_label=manual_label, manual_trusted=manual_trusted)
    result.data.update(selected_profile=profile, eligibility='review', reasons=[])
    if not config.enabled:
        result.status = Status.SKIPPED
        result.data['reasons'] = ['source_filter_disabled']
        return result
    if result.error or result.status != Status.SUCCESS or result.mock:
        result.data.update(eligibility='error', reasons=['source_analysis_error_or_incomplete'])
        return result
    try:
        values = Measurements.model_validate(result.data['measurements'])
    except Exception:
        result.data.update(eligibility='error', reasons=['source_measurements_invalid'])
        return result
    if values.face_count == 0:
        result.data.update(eligibility='ineligible', reasons=['no_face'])
        return result
    reasons = []
    if values.face_count is None or values.face_count != 1:
        reasons.append('single_face_not_confirmed')
    if result.data.get('conflicts'):
        reasons.append('observation_conflict')
    if result.data.get('quality_evidence', {}).get('status') != 'clear':
        reasons.append('quality_uncalibrated_or_unclear')
    nisp = result.data.get('observations', {}).get('nisp', {})
    if nisp.get('status') != 'success':
        reasons.append('nisp_unavailable_or_failed')
    group = 'mouth' if action.startswith('mouth_') else 'eyes' if action == 'eyes_open_to_closed' else 'frontal'
    cal = config.calibration.get(group, {})
    if cal.get('status') != 'validated' or not cal.get('evidence'):
        reasons.append(group + '_calibration_pending')
    if not manual_trusted or manual_label == 'unknown':
        reasons.append('manual_state_evidence_missing')
    if reasons:
        result.data['reasons'] = reasons
        return result
    required = {'mouth_closed_to_open':'mouth_closed', 'mouth_open_to_closed':'mouth_open',
                'eyes_open_to_closed':'eyes_open', 'frontal_to_left':'head_frontal', 'frontal_to_right':'head_frontal'}[action]
    if manual_label != required:
        result.data.update(eligibility='ineligible', reasons=['manual_state_mismatch'])
        return result
    try:
        keys = ['closed_max','open_min','nisp_mouth_open_min','nisp_mouth_closed_max'] if group=='mouth' else ['open_min','nisp_blink_max'] if group=='eyes' else ['max_abs_yaw','max_abs_pitch','nisp_frontal_min']
        if not all(type(cal[k]) in (int,float) and math.isfinite(cal[k]) and cal[k]>=0 for k in keys):
            raise ValueError('Invalid calibration numeric values')
        if group == 'mouth':
            closed, opened = float(cal['closed_max']), float(cal['open_min'])
            if not 0 <= closed < opened <= 2:
                raise ValueError()
            mouth_score=nisp['scores'].get('mouth')
            if (mouth_score is None or (action=='mouth_closed_to_open' and mouth_score>float(cal['nisp_mouth_closed_max']))
                    or (action=='mouth_open_to_closed' and mouth_score<float(cal['nisp_mouth_open_min']))):
                reasons.append('nisp_mouth_conflict_or_unknown')
            if values.lip_ratio is None:
                reasons.append('lip_ratio_unknown')
            elif action == 'mouth_closed_to_open' and values.lip_ratio > closed:
                reasons.append('manual_and_lip_ratio_conflict')
            elif action == 'mouth_open_to_closed' and values.lip_ratio < opened:
                reasons.append('manual_and_lip_ratio_conflict')
        elif group == 'eyes':
            opened = float(cal['open_min'])
            if not 0 < opened <= 2:
                raise ValueError()
            if values.eye_ratios is None or min(values.eye_ratios) < opened:
                reasons.append('both_eyes_not_clearly_open_or_conflict')
            if nisp['scores'].get('blink') is None or nisp['scores']['blink'] > float(cal['nisp_blink_max']):
                reasons.append('blink_conflict_or_unknown')
        else:
            if values.yaw_degrees is None or values.pitch_degrees is None:
                reasons.append('pose_unknown')
            elif abs(values.yaw_degrees) > float(cal['max_abs_yaw']) or abs(values.pitch_degrees) > float(cal['max_abs_pitch']):
                reasons.append('manual_and_pose_conflict')
            if nisp['scores'].get('frontal') is None or nisp['scores']['frontal'] < float(cal['nisp_frontal_min']):
                reasons.append('frontal_class_conflict_or_unknown')
    except (ValueError, KeyError, TypeError):
        reasons.append('calibration_parameters_invalid')
    result.data.update(eligibility='review' if reasons else 'eligible', reasons=reasons or ['all_required_calibrated_evidence_consistent'])
    return result


class PilotSourceRules(Contract):
    """Bounded visual Pilot policy, not validated production state calibration."""
    version: str = 'source_only_pilot30_v1_1028'
    enabled: bool = False
    scope: str = 'pilot_heuristic_not_production_calibrated'
    seed: int = 2026102809
    max_sources: int = Field(default=30, ge=1, le=30)
    closed_lip_max: float = Field(default=.03, ge=0, le=.1)
    open_lip_min: float = Field(default=.15, ge=.1, le=1)
    mouth_closed_score_max: float = Field(default=.2, ge=0, le=1)
    mouth_open_score_min: float = Field(default=.5, ge=0, le=1)
    eye_open_min: float = Field(default=.2, gt=0, le=1)
    eye_asymmetry_max: float = Field(default=.1, ge=0, le=1)
    blink_score_max: float = Field(default=.2, ge=0, le=1)
    frontal_yaw_max: float = Field(default=8, gt=0, le=25)
    frontal_pitch_max: float = Field(default=20, gt=0, le=30)
    frontal_roll_max: float = Field(default=15, gt=0, le=30)
    frontal_score_min: float = Field(default=.8, ge=0, le=1)
    side_score_max: float = Field(default=.2, ge=0, le=1)
    min_edge: int = Field(default=128, ge=32)
    brightness_min: float = 20
    brightness_max: float = 235
    contrast_min: float = 10
    laplacian_min: float = 10
    evidence: list[str] = Field(default_factory=list)


def pilot_basic_quality(data, config):
    """Reuse Examiner's descriptive pixel math, independent Pilot guard only."""
    from io import BytesIO
    import numpy as np
    from PIL import Image,ImageOps
    with Image.open(BytesIO(data)) as image:
        image=ImageOps.exif_transpose(image).convert('RGB')
        width,height=image.size
        image.thumbnail((512,512),Image.Resampling.LANCZOS)
        gray=np.asarray(image.convert('L'),dtype=np.float64)
    lap=gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]-4*gray[1:-1,1:-1]
    metrics=dict(width=width,height=height,brightness_mean=float(gray.mean()),contrast_std=float(gray.std()),laplacian_variance=float(lap.var()))
    reasons=[]
    if min(width,height)<config.min_edge:reasons.append('source_too_small_for_pilot')
    if not config.brightness_min<=metrics['brightness_mean']<=config.brightness_max:reasons.append('extreme_brightness')
    if metrics['contrast_std']<config.contrast_min:reasons.append('low_contrast')
    if metrics['laplacian_variance']<config.laplacian_min:reasons.append('severely_low_detail')
    return dict(status='review' if reasons else 'pilot_basic_checks_passed',reasons=reasons,metrics=metrics,
                occlusion='not_automatically_verified',semantic_quality='not_evaluated',scope=config.scope)


def assess_pilot_actions(analysis, *, config, quality):
    """Five action-specific decisions from real fields, retaining uncertainty."""
    result=analysis.model_copy(deep=True);result.module='source_rule_filter'
    result.data.update(eligibility='review',reasons=[],candidate_actions=[],action_decisions={},
                       rule_scope=config.scope,rule_version=config.version,quality_evidence=quality)
    def finish(state,reasons):
        result.data.update(eligibility=state,reasons=reasons)
        return result
    if not config.enabled:return finish('review',['pilot_filter_disabled'])
    if analysis.status!=Status.SUCCESS or analysis.error or analysis.mock:
        return finish('error',['source_analysis_failed_or_controlled'])
    try:
        m=Measurements.model_validate(analysis.data['measurements'])
    except (KeyError,ValueError,TypeError):return finish('error',['required_observations_missing_or_failed'])
    if m.face_count==0:return finish('ineligible',['no_face'])
    if m.face_count!=1:return finish('review',['single_face_not_confirmed'])
    if analysis.data.get('conflicts'):return finish('review',['independent_observations_conflict'])
    try:
        nisp=analysis.data['observations']['nisp'];s=nisp['scores']
        if nisp['status']!='success' or not all(type(s[k]) in (int,float) and math.isfinite(s[k]) and 0<=s[k]<=1 for k in ['mouth','blink','frontal','left','right']):
            raise ValueError('Incomplete NISP response')
    except (KeyError,ValueError,TypeError):return finish('error',['required_observations_missing_or_failed'])
    if quality.get('status')!='pilot_basic_checks_passed':return finish('review',quality.get('reasons') or ['pilot_basic_quality_unknown'])
    if any(v is None for v in [m.yaw_degrees,m.pitch_degrees,m.roll_degrees,m.lip_ratio,m.eye_ratios]):
        return finish('review',['continuous_measurements_missing'])
    views_ok=abs(m.yaw_degrees)<=25 and abs(m.pitch_degrees)<=30 and abs(m.roll_degrees)<=20
    frontal=(abs(m.yaw_degrees)<=config.frontal_yaw_max and abs(m.pitch_degrees)<=config.frontal_pitch_max
        and abs(m.roll_degrees)<=config.frontal_roll_max and s['frontal']>=config.frontal_score_min
        and max(s['left'],s['right'])<=config.side_score_max)
    matches={
        'mouth_closed_to_open':views_ok and m.lip_ratio<=config.closed_lip_max and s['mouth']<=config.mouth_closed_score_max,
        'mouth_open_to_closed':views_ok and m.lip_ratio>=config.open_lip_min and s['mouth']>=config.mouth_open_score_min,
        'eyes_open_to_closed':views_ok and min(m.eye_ratios)>=config.eye_open_min and abs(m.eye_ratios[0]-m.eye_ratios[1])<=config.eye_asymmetry_max and s['blink']<=config.blink_score_max,
        'frontal_to_left':frontal,'frontal_to_right':frontal,
    }
    reason={
        'mouth_closed_to_open':'lip_closed_and_aux_mouth_consistent' if matches['mouth_closed_to_open'] else 'closed_lip_or_mouth_aux_ambiguous_or_conflicting',
        'mouth_open_to_closed':'lip_open_and_aux_mouth_consistent' if matches['mouth_open_to_closed'] else 'open_lip_or_mouth_aux_ambiguous_or_conflicting',
        'eyes_open_to_closed':'both_eye_ratios_and_aux_blink_consistent' if matches['eyes_open_to_closed'] else 'eye_ratios_visibility_proxy_or_blink_conflict',
        'frontal_to_left':'pose_and_frontal_class_consistent' if frontal else 'pose_or_frontal_class_not_concordant',
        'frontal_to_right':'pose_and_frontal_class_consistent' if frontal else 'pose_or_frontal_class_not_concordant',
    }
    result.data['action_decisions']={a:dict(eligibility='eligible' if ok else 'review',reasons=[reason[a]],evidence_tier='pilot_concordant' if ok else 'uncertain') for a,ok in matches.items()}
    result.data['candidate_actions']=[a for a,ok in matches.items() if ok]
    result.data['state_estimates']=dict(mouth='pilot_closed' if matches['mouth_closed_to_open'] else 'pilot_open' if matches['mouth_open_to_closed'] else 'unknown',
        eyes='pilot_open' if matches['eyes_open_to_closed'] else 'unknown',pose='pilot_frontal' if frontal else 'unknown',
        eye_visibility='two_ratio_and_pose_proxy; no occlusion classifier',human_label='unknown unless separately reviewed')
    return finish('eligible' if result.data['candidate_actions'] else 'review',
                  ['pilot_consistent_evidence_not_production_calibration'] if result.data['candidate_actions'] else ['no_concordant_supported_action'])


class PilotActionScheduler:
    """Balance only concordant actions, then intensity; stable seed/SHA ties."""
    def __init__(self, seed=2026102809):
        from collections import Counter
        self.seed=seed;self.actions=Counter();self.levels=Counter();self.seen=set();self.last_selection=None

    def select(self, decision):
        import hashlib
        data=decision.data;source=data['source_sha256']
        if data['eligibility']!='eligible' or source in self.seen:return None
        actions=data['candidate_actions']
        if not actions:return None
        def tie(value):return hashlib.sha256((str(self.seed)+':'+source+':'+value).encode()).hexdigest()
        action=min(actions,key=lambda a:(self.actions[a],tie(a)))
        levels=['canonical'] if action in ['mouth_open_to_closed','eyes_open_to_closed'] else ['L1','L2','L3']
        level=min(levels,key=lambda l:(self.levels[action,l],tie(action+':'+l)))
        self.last_selection=dict(seed=self.seed,method='equal-evidence least-used-action then least-used-level; SHA tie',
            candidates=list(actions),action_counts_before=dict(self.actions),selected_action=action,selected_level=level)
        self.actions[action]+=1;self.levels[action,level]+=1;self.seen.add(source)
        return select_profile(action,level)
