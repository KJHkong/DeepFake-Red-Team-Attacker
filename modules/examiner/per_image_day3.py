"""Independent per-image validity checks. No Generator, retrieval, Detector RPC or Gates."""
import hashlib
from io import BytesIO
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import PIL
from PIL import Image, ImageOps

from shared.contracts import ArtifactAccess, FaceObservation, FacePresenceDetector, ModuleFailure
from shared.schemas import (AttackRequest, Candidate, ErrorInfo, ExaminerCheck, ExaminerResult,
                            ModuleResult, ReferenceRecord, Status, now)


class PerImageExaminer:
    def __init__(self,artifact_access:ArtifactAccess,*,face_detector:FacePresenceDetector|None=None):
        self.access=artifact_access;self.face_detector=face_detector
        self.code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def examine(self,request:AttackRequest,candidate:Candidate,references:list[ReferenceRecord])->ExaminerResult:
        started=perf_counter();artifact=candidate.artifact
        checks={name:ExaminerCheck(name=name,required=name in {'artifact_access','artifact_integrity','image_decode','image_safety','dimensions'})
            for name in ['artifact_access','artifact_integrity','image_decode','image_safety','dimensions','face_presence',
                         'basic_quality','task_semantics','business_distribution','identity_preservation','deepfake_authenticity']}
        checks['face_presence'].required=request.constraints.require_face
        checks['task_semantics'].required=bool(request.constraints.required_capabilities)
        for name in ['business_distribution','identity_preservation','deepfake_authenticity']:
            checks[name].reason='no_reliable_evaluator_in_this_phase'
        checks['task_semantics'].reason='requested_semantic_capabilities_not_pixel_verified' if request.constraints.required_capabilities else 'no_additional_pixel_verifiable_rule_declared'
        checks['task_semantics'].metrics={'requested_capabilities':list(request.constraints.required_capabilities)}
        info={};metrics={};face_present=None;execution=Status.SUCCESS;error=None

        def set_check(name,status,*,reason=None,values=None,elapsed=0):
            check=checks[name];check.status=status;check.reason=reason;check.metrics=values or {};check.elapsed_ms=elapsed

        def finish():
            required=[c for c in checks.values() if c.required]
            def decision(values):
                if any(c.status=='failed' for c in values):return False
                return True if values and all(c.status=='passed' for c in values) else None
            validity=decision(required)
            constraint=decision([checks[n] for n in ['dimensions','face_presence','task_semantics'] if checks[n].required])
            if execution!=Status.SUCCESS:validity=None;constraint=None
            reasons=[f'{c.name}:{c.reason or c.status}' for c in checks.values()
                if c.status in {'failed','error','unavailable'} or (c.required and c.status=='not_evaluated')]
            return ExaminerResult(candidate_id=candidate.candidate_id,artifact_id=artifact.artifact_id,
                source_sha256=artifact.sha256,validity_passed=validity,constraints_passed=constraint,
                distribution_check='not_evaluated',gate1_passed=None,face_present=face_present,
                reference_ids=[r.sample_id for r in references],reasons=reasons,metrics=metrics,mock=candidate.mock,
                assessment_scope='per_image',execution_status=execution,checks=list(checks.values()),image_info=info,
                elapsed_ms=(perf_counter()-started)*1000,error=error,
                provenance={'examiner_version':'per_image_v0.1','code_sha256':self.code_sha256,
                    'runtime':{'python':platform.python_version(),'pillow':PIL.__version__,'numpy':np.__version__},
                    'input_policy':{'max_file_bytes':self.access.policy.max_file_bytes,'max_pixels':self.access.policy.max_pixels,
                        'allowed_formats':self.access.policy.allowed_formats},
                    'declared_metadata':{'sha256':artifact.sha256,'width':artifact.width,'height':artifact.height,
                        'media_type':artifact.media_type,'size_bytes':artifact.size_bytes,'frame_count':artifact.frame_count},
                    'quality_method':'exif_transpose; RGB; thumbnail maxedge512 LANCZOS; Pillow L; numpy float64; 4-neighbor interior Laplacian',
                    'quality_thresholds_used':False,'reference_pixels_read':False,'reference_context_used_for_distribution':False,
                    'task':{'generator':request.generator,'taxonomy':request.taxonomy.value,'require_face':request.constraints.require_face,
                        'min_width':request.constraints.min_width,'min_height':request.constraints.min_height}})

        if candidate.mock!=artifact.mock:
            set_check('artifact_integrity','failed',reason='candidate_artifact_mock_provenance_mismatch')
            return finish()
        tick=perf_counter()
        try:raw=self.access.read(artifact)
        except ModuleFailure as exc:
            if exc.code in {'path_not_allowed','permission_denied','invalid_path'}:
                execution=Status.INVALID_INPUT;error=ErrorInfo(code=exc.code,message='Artifact access was denied by the server policy')
                set_check('artifact_access','error',reason=exc.code,elapsed=(perf_counter()-tick)*1000)
            elif exc.code in {'invalid_artifact','input_changed','invalid_image','image_too_large','file_too_large','file_not_found','invalid_file'}:
                name={'invalid_artifact':'artifact_integrity','input_changed':'artifact_integrity','invalid_image':'image_decode',
                    'image_too_large':'image_safety','file_too_large':'image_safety'}.get(exc.code,'artifact_access')
                set_check(name,'failed',reason=exc.code,elapsed=(perf_counter()-tick)*1000)
            else:
                execution=Status.MODULE_FAILED;error=ErrorInfo(code='artifact_gateway_failed',message='Artifact gateway did not complete')
                set_check('artifact_access','error',reason='artifact_gateway_failed',elapsed=(perf_counter()-tick)*1000)
            return finish()
        except Exception:
            execution=Status.MODULE_FAILED;error=ErrorInfo(code='artifact_gateway_failed',message='Artifact gateway did not complete')
            set_check('artifact_access','error',reason='artifact_gateway_failed',elapsed=(perf_counter()-tick)*1000)
            return finish()
        if not isinstance(raw,bytes):
            execution=Status.MODULE_FAILED;error=ErrorInfo(code='invalid_gateway_result',message='Artifact gateway returned an invalid result')
            set_check('artifact_access','error',reason='invalid_gateway_result',elapsed=(perf_counter()-tick)*1000)
            return finish()
        set_check('artifact_access','passed',elapsed=(perf_counter()-tick)*1000)
        tick=perf_counter()
        try:
            if not isinstance(raw,bytes) or not 0<len(raw)<=self.access.policy.max_file_bytes:
                raise ModuleFailure('file_too_large','Bounded image bytes required')
            with Image.open(BytesIO(raw),formats=self.access.policy.allowed_formats) as image:
                width,height=image.size;fmt=image.format;frames=getattr(image,'n_frames',1)
                if width*height>self.access.policy.max_pixels:raise ModuleFailure('image_too_large','Pixel limit exceeded')
                if frames!=1:raise ModuleFailure('invalid_image','Single frame required')
                image.load()
                info.update(width=width,height=height,format=fmt,pixels=width*height,frame_count=frames,size_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest())
                rgb=ImageOps.exif_transpose(image).convert('RGB')
            set_check('image_decode','passed',values={'format':fmt,'single_frame':True},elapsed=(perf_counter()-tick)*1000)
            set_check('image_safety','passed',values={'pixels':width*height,'size_bytes':len(raw)})
            media={'PNG':'image/png','JPEG':'image/jpeg','WEBP':'image/webp'}[fmt]
            matches=(artifact.sha256==info['sha256'] and artifact.media_type==media and
                (artifact.width is None or artifact.width==width) and (artifact.height is None or artifact.height==height) and
                (artifact.size_bytes is None or artifact.size_bytes==len(raw)) and (artifact.frame_count is None or artifact.frame_count==frames))
            compared=['sha256','media_type']+[n for n in ['width','height','size_bytes','frame_count'] if getattr(artifact,n) is not None]
            missing=[n for n in ['width','height','size_bytes','frame_count'] if getattr(artifact,n) is None]
            set_check('artifact_integrity','passed' if matches else 'failed',reason=None if matches else 'artifact_metadata_mismatch',
                values={'compared_fields':compared,'optional_metadata_absent':missing,'sha256_verified':artifact.sha256==info['sha256']})
            size_ok=width>=request.constraints.min_width and height>=request.constraints.min_height
            set_check('dimensions','passed' if size_ok else 'failed',reason=None if size_ok else 'below_requested_minimum',
                values={'width':width,'height':height,'minimum_width':request.constraints.min_width,'minimum_height':request.constraints.min_height})
        except ModuleFailure as exc:
            set_check('image_safety','failed',reason=exc.code)
            return finish()
        except (OSError,ValueError,Image.DecompressionBombError):
            set_check('image_decode','failed',reason='invalid_image')
            return finish()
        except Exception:
            execution=Status.MODULE_FAILED;error=ErrorInfo(code='image_check_failed',message='Image diagnostics did not complete')
            set_check('image_decode','error',reason='image_check_failed')
            return finish()
        tick=perf_counter()
        try:
            rgb.thumbnail((512,512),Image.Resampling.LANCZOS)
            gray=np.asarray(rgb.convert('L'),dtype=np.float64)
            lap=gray[1:-1,:-2]+gray[1:-1,2:]+gray[:-2,1:-1]+gray[2:,1:-1]-4*gray[1:-1,1:-1]
            metrics.update(brightness_mean=float(gray.mean()),contrast_std=float(gray.std()),
                laplacian_variance=float(lap.var()) if lap.size else None)
            set_check('basic_quality','measured',reason='diagnostic_only_no_acceptance_thresholds',
                values={**metrics,'thumbnail_width':gray.shape[1],'thumbnail_height':gray.shape[0]},elapsed=(perf_counter()-tick)*1000)
        except Exception:
            execution=Status.MODULE_FAILED;error=ErrorInfo(code='quality_check_failed',message='Quality diagnostics did not complete')
            set_check('basic_quality','error',reason='quality_check_failed',elapsed=(perf_counter()-tick)*1000)
            return finish()
        if any(c.required and c.status=='failed' for c in checks.values()):
            checks['face_presence'].reason='not_run_after_hard_image_failure'
            return finish()
        if self.face_detector is None:
            set_check('face_presence','unavailable' if request.constraints.require_face else 'not_evaluated',reason='face_detector_not_configured')
            return finish()
        tick=perf_counter()
        try:
            observation=self.face_detector.detect(artifact,raw)
            if (not isinstance(observation,FaceObservation) or observation.source_sha256!=info['sha256'] or
                    type(observation.face_count) is not int or not 0<=observation.face_count<=10000 or
                    not math.isfinite(observation.elapsed_ms) or observation.elapsed_ms<0 or (observation.mock and not candidate.mock)):
                raise ModuleFailure('invalid_face_result','Face observation cannot be used as real bound evidence')
            face_present=observation.face_count>0
            status=('passed' if face_present else 'failed') if request.constraints.require_face else 'measured'
            set_check('face_presence',status,reason='no_face_detected' if not face_present else None,
                values={'face_count':observation.face_count,'method':observation.method,'runtime':observation.runtime,
                    'provider_mock':observation.mock,'provider_elapsed_ms':observation.elapsed_ms,'details':observation.details},
                elapsed=(perf_counter()-tick)*1000)
        except Exception as exc:
            code=exc.code if isinstance(exc,ModuleFailure) else 'face_detector_failed'
            execution=Status.MODULE_FAILED;error=ErrorInfo(code=code,message='Face presence check did not complete')
            set_check('face_presence','error',reason=code,elapsed=(perf_counter()-tick)*1000)
        return finish()

    def invoke(self,request,candidate,references=None)->ModuleResult:
        stamp=now();result=self.examine(request,candidate,references or [])
        return ModuleResult(module='examiner',status=result.execution_status,mock=result.mock,started_at=stamp,
            elapsed_ms=result.elapsed_ms,inputs=[candidate.candidate_id,candidate.artifact.artifact_id],outputs=[],
            parameters={'constraints':request.constraints.model_dump(mode='json')},
            data={'examiner_result':result.model_dump(mode='json')},error=result.error)
