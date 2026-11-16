"""Current scale policy: five actions only; historical Pilot banks are retained."""
from collections import Counter
import hashlib
import random
from pydantic import Field
from modules.source_state.day10 import Day10Rules,assess_day10_actions,select_day10_profile

SCALE_ACTIONS=('mouth_closed_to_open','mouth_open_to_closed','eyes_open_to_closed','frontal_to_left','frontal_to_right')


class ScaleRules(Day10Rules):
    version: str='scale_five_actions_v1_1029'
    seed: int=2026102910
    max_sources: int=Field(default=100,ge=1,le=100)


def select_scale_profile(action,level=None):
    if action not in SCALE_ACTIONS:raise ValueError('Action removed from the current scale policy')
    return select_day10_profile(action,level)


def assess_scale_actions(analysis,*,config,quality):
    result=assess_day10_actions(analysis,config=config,quality=quality)
    result.data['candidate_actions']=[a for a in result.data['candidate_actions'] if a in SCALE_ACTIONS]
    result.data['action_decisions']={k:v for k,v in result.data.get('action_decisions',{}).items() if k in SCALE_ACTIONS}
    result.data['action_policy']='five_actions_no_keep_frontal'
    if result.data['eligibility']=='eligible' and not result.data['candidate_actions']:
        result.data.update(eligibility='review',reasons=['no_supported_scale_action'])
    return result


class ScaleScheduler:
    """Balance five supported actions, then levels; never fill unsupported quotas."""
    def __init__(self,seed=2026102910):
        self.seed=seed;self.actions=Counter();self.levels=Counter();self.seen=set();self.last_selection=None
    def planning_key(self,task):
        acts=task['source_decision']['data']['candidate_actions']
        return len(acts),hashlib.sha256((str(self.seed)+':'+task['task_id']).encode()).hexdigest()
    def select(self,decision):
        data=decision.data;sha=data['source_sha256']
        actions=[a for a in data.get('candidate_actions',[]) if a in SCALE_ACTIONS]
        if data['eligibility']!='eligible' or sha in self.seen or not actions:return None
        tie=lambda value:hashlib.sha256((str(self.seed)+':'+sha+':'+value).encode()).hexdigest()
        action=min(actions,key=lambda a:(self.actions[a],tie(a)))
        levels=['canonical'] if action in ('mouth_open_to_closed','eyes_open_to_closed') else ['L1','L2','L3']
        level=min(levels,key=lambda l:(self.levels[action,l],tie(action+':'+l)))
        self.last_selection=dict(seed=self.seed,method='least-used compatible action, then least-used level; SHA tie',
            candidates=actions,action_counts_before=dict(self.actions),selected_action=action,selected_level=level)
        self.seen.add(sha);self.actions[action]+=1;self.levels[action,level]+=1
        return select_scale_profile(action,level)


def sample_roles(rows,*,swapping,editing,seed):
    if type(swapping) is not int or type(editing) is not int or swapping<0 or editing<0:raise ValueError('Nonnegative quotas required')
    # Stable representatives for byte aliases, without modifying or deleting inputs.
    unique={}
    for row in sorted(rows,key=lambda r:str(r['path'])):unique.setdefault(row['sha256'],row)
    pool=list(unique.values());required=swapping*2+editing
    if required>len(pool):raise ValueError(f'Requested {required} unique images; available {len(pool)}')
    rng=random.Random(seed);source=rng.sample(pool,swapping);seen={r['sha256'] for r in source}
    remaining=[r for r in pool if r['sha256'] not in seen]
    target=rng.sample(remaining,swapping);seen.update(r['sha256'] for r in target)
    remaining=[r for r in remaining if r['sha256'] not in seen]
    edit=rng.sample(remaining,editing)
    return dict(seed=seed,swapping_source=source,swapping_target=target,editing_source=edit,
                unique_available=len(pool),required=required,role_overlap=False,pairing='zip in sampled source/target order')
