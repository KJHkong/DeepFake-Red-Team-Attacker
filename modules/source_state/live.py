"""Compose fresh independent observations into the existing SourceStateAnalyzer.

No state is inferred from SHA, generated images or an unavailable classifier.
"""
from dataclasses import asdict, is_dataclass
from time import perf_counter
from modules.source_state.analyzer import SourceStateAnalyzer


def _unavailable_nisp():
    return dict(status='not_evaluated', scores={}, raw_response=None,
                capabilities={k: 'unsupported' for k in ['face_count', 'yaw_degrees', 'pitch_degrees', 'roll_degrees', 'eye_ratios', 'lip_ratio']},
                reason='NISP backend not configured; no replacement score')


class LiveSourceAnalyzer:
    def __init__(self, *, face_call, liveportrait_call, nisp_call=None, provenance=None, allow_controlled_inputs=False,
                 face_count_policy='agreement'):
        if face_count_policy not in ('agreement','controlled_source_or'):
            raise ValueError('Unknown Source count policy')
        self.face_count_policy=face_count_policy
        self.face_call, self.lp_call, self.nisp_call = face_call, liveportrait_call, nisp_call
        self.provenance = dict(provenance or {})
        self.allow_controlled_inputs = allow_controlled_inputs

    def analyze(self, artifact, data):
        observations = dict(nisp=_unavailable_nisp(),face_count=dict(status='not_evaluated'),liveportrait=dict(status='not_evaluated'))
        conflicts, missing = [], []
        def collect(a, raw):
            start=perf_counter()
            try:
                face = self.face_call(a, raw)
            except Exception as error:
                observations['face_count']=dict(status='error',error=dict(type=type(error).__name__,code=getattr(error,'code',None)),elapsed_ms=(perf_counter()-start)*1000)
                raise
            face_row = asdict(face) if is_dataclass(face) else vars(face)
            count = face.face_count
            if type(count) is not int or not 0 <= count <= 10000:
                raise ValueError('Invalid independent face count')
            observations['face_count'] = dict(face_row, status='success', count=count)
            start=perf_counter()
            try:
                lp = self.lp_call(a, raw, count)
            except Exception as error:
                observations['liveportrait']=dict(status='error',error=dict(type=type(error).__name__,code=getattr(error,'code',None)),elapsed_ms=(perf_counter()-start)*1000)
                raise
            observations['liveportrait'] = lp
            if lp.get('source_sha256') != a.sha256 or lp.get('status') != 'success':
                raise ValueError('LivePortrait result failed or changed Source binding')
            if type(lp.get('face_count')) is not int:
                raise ValueError('Missing independent LivePortrait face count')
            single=(count==1 or lp['face_count']==1) if self.face_count_policy=='controlled_source_or' else count==lp['face_count']==1
            if count != lp['face_count'] and self.face_count_policy=='agreement':
                conflicts.append('face_count_disagreement')
            if self.nisp_call:
                start=perf_counter()
                try:
                    observations['nisp'] = self.nisp_call(a, raw)
                except Exception as error:
                    observations['nisp']=dict(status='error',error=dict(type=type(error).__name__,code=getattr(error,'code',None)),elapsed_ms=(perf_counter()-start)*1000)
                    raise
                nisp = observations['nisp']
                if nisp.get('source_sha256') != a.sha256:
                    raise ValueError('NISP Source binding mismatch')
                if nisp.get('status') == 'no_face' and count > 0:
                    conflicts.append('nisp_no_face_but_other_detector_has_face')
            if observations['nisp']['status'] == 'not_evaluated':
                missing.append('nisp_not_evaluated')
            elif observations['nisp']['status'] == 'error':
                missing.append('nisp_backend_error')
            combined_count=1 if single else count if count==lp['face_count'] else None
            measurements = dict(face_count=combined_count,
                                quality='unknown', confidence=None,
                                measurement_semantics='CPU landmark/motion estimates; independent HOG count; no calibrated state probability')
            for key in ['yaw_degrees', 'pitch_degrees', 'roll_degrees', 'eye_ratios', 'lip_ratio']:
                measurements[key] = lp.get(key) if single else None
                if measurements[key] is None:
                    missing.append(key + '_unknown')
            return measurements
        validation_artifact = artifact.model_copy(update={'mock':False}) if artifact.mock and self.allow_controlled_inputs else artifact
        result = SourceStateAnalyzer(collect, backend='liveportrait_analyze_only_with_independent_hog',
                                     provenance=self.provenance).analyze(validation_artifact, data)
        result.mock = artifact.mock
        result.data.update(source_artifact_id=artifact.artifact_id,
                           input_integrity='failed' if result.error and result.error.code == 'source_input_changed' else 'passed',
                           observations=observations, conflicts=conflicts, missing=missing,face_count_policy=self.face_count_policy,
                           observation_status='error' if result.error else 'review' if conflicts or missing else 'complete',
                           quality_evidence=dict(status='unknown', reason='No calibrated automatic occlusion/blur acceptance'),
                           eligibility='review', default_filter_enabled=False, controlled_input=artifact.mock)
        return result
