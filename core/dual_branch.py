"""Unified internal dataset-task entry, composing the existing content engine.

Roles route the task; no client Prompt, model path or production Gate overrides.
Batch planning and one-task run share prepare/execute; no duplicate generator loop.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from uuid import uuid4

from core.artifacts import LocalArtifactAccess
from core.inputs import LocalImageIngestor
from core.editing_source import build_editing_executor
from core.orchestrator import Orchestrator
from modules.source_state.rules import pilot_basic_quality
from modules.source_state.day10 import assess_day10_actions,Day10Scheduler,select_day10_profile
from shared.schemas import AttackRequest,InputAsset,InputPolicy,ModuleResult,Status

TOOLS=('inswapper','simswap','blendface')
MILD_OPERATORS=(
    ('jpeg_recompress',dict(quality=95,subsampling=0)),('gaussian_blur',dict(radius=.2)),
    ('color_adjust',dict(brightness=1.02,contrast=1.,color=1.)),('gaussian_noise',dict(std=.5,seed=20261029)),
    ('resize_resample',dict(scale=.95,resample='lanczos')),('sharpen',dict(radius=.5,percent=25,threshold=3)),
    ('crop_pad',dict(crop_box=None,padding=[1,1,1,1],fill_mode='edge',fill=[0,0,0])),
    ('pixelate',dict(scale=.75,down_resample='box',up_resample='nearest')))


def route_inputs(inputs):
    roles=Counter(a.role for a in inputs)
    if roles=={'source':1}:return 'editing'
    if roles=={'source':1,'target':1}:return 'swapping'
    raise ValueError('Expected one source, or exactly one source and one target; Prompt is automatic')


def fixed_batch_schedule(swapping,editing,*,seed=2026102901):
    return balanced_batch_schedule(swapping,editing,seed=seed,limits=(15,15))


def balanced_batch_schedule(swapping,editing,*,seed,limits):
    if not 0<=swapping<=limits[0] or not 0<=editing<=limits[1]:raise ValueError('Batch exceeds server-owned limits')
    rng=random.Random(seed);tools=[TOOLS[i%3] for i in range(swapping)];rng.shuffle(tools)
    rows=[dict(branch='swapping',index=i,generator=t,modifier=None) for i,t in enumerate(tools)]
    rows += [dict(branch='editing',index=i,generator='liveportrait_static',modifier=None) for i in range(editing)]
    selected=[i for i,r in enumerate(rows) if (r['index']%2==0 if r['branch']=='swapping' else r['index']%2==1)]
    rng.shuffle(selected)
    for index,slot in enumerate(selected):
        tool,params=MILD_OPERATORS[index%len(MILD_OPERATORS)]
        rows[slot]['modifier']=dict(tool_name=tool,parameters=dict(params))
    return rows


def write_new(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


class DualBranchPipeline:
    """Server-owned dependencies and scope; no public full_pipeline admission."""
    def __init__(self,root,*,input_policy,analyzer,rules,face_detector,detector,swapping_registry_factory,gpu_guard,
                 source_assessor=assess_day10_actions,profile_resolver=select_day10_profile,
                 scheduler_factory=Day10Scheduler,schedule_factory=None):
        self.root=Path(root);self.policy=input_policy;self.analyzer=analyzer;self.rules=rules
        self.face=face_detector;self.detector=detector;self.registry_factory=swapping_registry_factory;self.gpu_guard=gpu_guard
        self.scheduler=scheduler_factory(rules.seed);self.single_tools=Counter()
        self.source_assessor=source_assessor;self.profile_resolver=profile_resolver
        self.schedule_factory=schedule_factory or (lambda s,e:fixed_batch_schedule(s,e,seed=rules.seed))
        self.root.mkdir(parents=True,exist_ok=True)

    def prepare(self,inputs,*,task_id=None,expected_sha_by_role=None):
        assets=[InputAsset.model_validate(a) for a in inputs];branch=route_inputs(assets)
        task_id=task_id or 'dataset_task_'+uuid4().hex+'_1029'
        if not task_id.replace('_','').isalnum():raise ValueError('Invalid task ID')
        folder=self.root/task_id;folder.mkdir(mode=0o700)
        start=perf_counter();snapshots=LocalImageIngestor(self.policy).ingest(assets,folder/'inputs')
        if expected_sha_by_role is not None and {a.role:a.sha256 for a in snapshots}!=expected_sha_by_role:
            raise ValueError('Frozen sampling input SHA changed before Source analysis')
        access=LocalArtifactAccess(InputPolicy(allowed_roots=[str(folder)]),output_root=folder)
        task=dict(task_id=task_id,task_record_id=uuid4().hex,branch=branch,folder=str(folder),
            input_artifacts=[a.model_dump(mode='json') for a in snapshots],status='prepared',source_analysis=None,
            source_decision=None,profile=None,generated_analyzer_calls=0,prepared_at=datetime.now(timezone.utc).isoformat())
        if branch=='editing':
            source=next(a for a in snapshots if a.role=='source');raw=access.read(source)
            analysis=self.analyzer.analyze(source,raw);quality=pilot_basic_quality(raw,self.rules)
            decision=self.source_assessor(analysis,config=self.rules,quality=quality)
            task.update(source_analysis=analysis.model_dump(mode='json'),source_decision=decision.model_dump(mode='json'),quality=quality)
        task['preparation_elapsed_ms']=(perf_counter()-start)*1000
        write_new(folder/'prepared_source_1029.json',task)
        return task

    def plan_batch(self,tasks):
        by_branch={b:[t for t in tasks if t['branch']==b] for b in ('swapping','editing')}
        schedule=self.schedule_factory(len(by_branch['swapping']),len(by_branch['editing']))
        for slot in schedule:by_branch[slot['branch']][slot['index']]['execution_plan']=slot
        # Scarce eligible action families first, deterministic SHA ties; no score use.
        eligible=[t for t in by_branch['editing'] if t['source_decision']['data']['eligibility']=='eligible']
        family=lambda a:'mouth' if a.startswith('mouth') else 'eyes' if a.startswith('eyes') else 'pose'
        default_key=lambda t:(len({family(a) for a in t['source_decision']['data']['candidate_actions']}),
            hashlib.sha256((str(self.rules.seed)+t['task_id']).encode()).hexdigest())
        eligible.sort(key=getattr(self.scheduler,'planning_key',default_key))
        for task in eligible:
            profile=self.scheduler.select(ModuleResult.model_validate(task['source_decision']))
            task.update(profile=profile,selection=self.scheduler.last_selection)
        for task in tasks:
            write_new(Path(task['folder'])/'execution_plan_1029.json',task)
        return tasks

    def run(self,inputs,*,task_id=None):
        """Unified one-task entry; Source-only callers never supply a Prompt."""
        task=self.prepare(inputs,task_id=task_id)
        if task['branch']=='editing':
            task['profile']=self.scheduler.select(ModuleResult.model_validate(task['source_decision']))
            task['selection']=self.scheduler.last_selection
            tool='liveportrait_static'
        else:
            tool=min(TOOLS,key=lambda t:(self.single_tools[t],TOOLS.index(t)));self.single_tools[tool]+=1
        task['execution_plan']=dict(branch=task['branch'],generator=tool,modifier=None)
        write_new(Path(task['folder'])/'execution_plan_1029.json',task)
        return self.execute(task)

    def execute(self,task):
        from shared.schemas import Artifact
        start=perf_counter();folder=Path(task['folder']);write_new(folder/'execution_attempt_1029.json',dict(attempt=1,task_id=task['task_id']))
        result=dict(task_id=task['task_id'],task_record_id=task['task_record_id'],branch=task['branch'],status='Error',
            run_id=None,generated=None,modified=None,final_candidate=None,gate1='not_evaluated',detector=None,
            task=task,generated_analyzer_calls=0,worker_elapsed_ms=None,execution_errors=[],attack_success=None)
        try:
            access=LocalArtifactAccess(InputPolicy(allowed_roots=[str(folder)]),output_root=folder)
            inputs=[Artifact.model_validate(a) for a in task['input_artifacts']]
            for a in inputs:access.read(a)
            plan=task['execution_plan'];modifier=plan['modifier'];profile=task['profile']
            if task['branch']=='editing':
                decision=task['source_decision']['data']
                if decision['eligibility']!='eligible' or not profile:
                    result['status']=decision['eligibility'].capitalize()
                    return result
                source=next(a for a in inputs if a.role=='source')
                if task['source_analysis']['data']['source_sha256']!=source.sha256:raise ValueError('Analyzed Source changed')
                if profile['action'] not in decision['candidate_actions'] or profile!=self.profile_resolver(profile['action'],profile['intensity_level']):raise ValueError('Selected profile changed')
                source=source.model_copy(update={'role':'image'})
                execute=build_editing_executor(profile,source,runs_root=folder/'runs',input_policy=InputPolicy(allowed_roots=[str(folder/'inputs')]),face_detector=self.face,profile_resolver=self.profile_resolver)
                engine=execute.func.__self__ if hasattr(execute,'func') else execute.__self__
                request=AttackRequest(mode='real',execution_scope='content_only',generator='liveportrait_static',taxonomy='Deepfake/Face_Editing',
                    inputs=[dict(role='image',path=source.path,display_allowed=source.display_allowed)],prompt=profile['prompt_text'],parameters=profile['parameters'],budget=dict(max_duration_s=240))
            else:
                tool=plan['generator']
                if tool not in TOOLS:raise ValueError('Swapping tool outside frozen pool')
                registry=self.registry_factory(tool)
                # Snapshot-bound ingestion uses existing canonical reader and SHA checks.
                ingestor=LocalImageIngestor(InputPolicy(allowed_roots=[str(folder/'inputs')]))
                expected={a.role:(a.path,a.sha256) for a in inputs}
                class BoundInputs:
                    def ingest(self,assets,destination):
                        if {a.role:(a.path) for a in assets}!={k:v[0] for k,v in expected.items()}:raise ValueError('Bound inputs changed')
                        snapshots=ingestor.ingest(assets,destination)
                        if {a.role:a.sha256 for a in snapshots}!={k:v[1] for k,v in expected.items()}:raise ValueError('Bound SHA changed')
                        return snapshots
                engine=Orchestrator(folder/'runs',registry=registry,input_ingestor=BoundInputs(),output_topic='dataset_swapping')
                engine.accepted_generators=engine.accepted_generators|{tool}
                request=AttackRequest(mode='real',execution_scope='content_only',generator=tool,taxonomy='Deepfake/Face_Swapping',inputs=[dict(role=a.role,path=a.path,display_allowed=a.display_allowed) for a in inputs],budget=dict(max_duration_s=240))
                execute=lambda r:engine.run_examined(r,face_detector=self.face)
            if modifier:
                from shared.schemas import ModifierRequest,ModifierOperation
                request.modifier=ModifierRequest(enabled=True,chain=[ModifierOperation.model_validate(modifier)])
            request.constraints.allow_business_display=all(a.display_allowed for a in inputs)
            if not self.gpu_guard():raise RuntimeError('GPU1 busy; no reallocation or retries')
            run,assessment=execute(request);directory=engine.store._directory(run.run_id)
            generated=next((c for c in run.candidates if c.artifact.stage=='generator'),None)
            final=next((c for c in run.candidates if c.candidate_id==run.final_candidate_id),None) if run.state=='content_generated' else None
            result.update(run_id=run.run_id,run_state=run.state,metadata_path=str(directory/'metadata.json'),assessment_path=assessment.metadata_path if assessment else None,
                generated=generated.model_dump(mode='json') if generated else None,final_candidate=final.model_dump(mode='json') if final else None,
                modified=final.model_dump(mode='json') if final and modifier else None,gate1=assessment.gate.decision if assessment else 'not_evaluated',
                worker_elapsed_ms=generated.metadata.get('worker_elapsed_ms') if generated else None,execution_errors=[e.model_dump(mode='json') for e in run.errors])
            write_new(directory/'dual_branch_task_1029.json',dict(task=task,run_id=run.run_id,generated_analyzer_calls=0))
            if final and assessment and assessment.gate.decision=='passed':
                access.read(final.artifact);score=self.detector.infer_one(final.artifact)
                result['detector']=score.model_dump(mode='json');write_new(directory/'official_detector_1029.json',result['detector'])
            result['status']='Generated' if final else 'Error'
        except Exception as exc:
            result.update(status='Error',error=dict(type=type(exc).__name__,code=getattr(exc,'code',None)))
        finally:
            result['execution_elapsed_ms']=(perf_counter()-start)*1000
            result['total_task_elapsed_ms']=task['preparation_elapsed_ms']+result['execution_elapsed_ms']
            write_new(folder/'result_1029.json',result)
        return result
