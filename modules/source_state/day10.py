"""Controlled-Source OR policy, action-family balancing and pose fallback."""
import hashlib
import math
from collections import Counter
from pydantic import Field
from shared.schemas import Status
from modules.source_state.rules import PilotSourceRules,select_profile

FAMILIES={'mouth_closed_to_open':'mouth','mouth_open_to_closed':'mouth','eyes_open_to_closed':'eyes',
          'frontal_to_left':'pose','keep_frontal':'pose','frontal_to_right':'pose'}


class Day10Rules(PilotSourceRules):
    version: str='controlled_dual_branch_v1_1029'
    seed: int=2026102901
    controlled_sources: bool=True
    face_count_policy: str='controlled_source_or'
    fallback_frontal_score_min: float=Field(default=.5,ge=0,le=1)


def select_day10_profile(action,level=None):
    if action!='keep_frontal':
        p=select_profile(action,level)
    else:
        from modules.generator.adapters.day10_actions import BANK,VERSION,resolve,bank_sha
        import json
        e=json.loads(BANK.read_text())['entry'];resolve(e['prompt_id'],e['prompt_text'])
        if level not in (None,'canonical'):raise ValueError('Keep-frontal has one target')
        p=dict(e,worker_bank_version=VERSION,bank_sha256=bank_sha(),frozen_profile_version=VERSION,frozen_config_sha256=bank_sha(),
            parameters=dict(prompt_id=e['prompt_id'],bank_version=VERSION),native_text_conditioning=False)
    return dict(p,action_family=FAMILIES[action])


def assess_day10_actions(analysis,*,config,quality):
    r=analysis.model_copy(deep=True);r.module='source_rule_filter'
    r.data.update(rule_version=config.version,rule_scope=config.scope,eligibility='review',reasons=[],candidate_actions=[],fallback_pose=False)
    def end(state,reason):r.data.update(eligibility=state,reasons=[reason]);return r
    if not config.enabled or not config.controlled_sources or config.face_count_policy!='controlled_source_or':return end('review','controlled_pilot_not_enabled')
    if r.error or r.status!=Status.SUCCESS or r.mock:return end('error','source_analysis_failed')
    try:
        obs=r.data['observations'];hog=obs['face_count']['count'];lp=obs['liveportrait']['face_count']
        if type(hog) is not int or type(lp) is not int:raise ValueError()
        single=hog==1 or lp==1
        r.data['single_face_policy']=dict(policy='HOG==1 OR LP==1',hog=hog,lp=lp,confirmed=single,assumption='controlled input defaults to single-face')
        if not single:return end('ineligible' if hog==lp==0 else 'review','neither_backend_confirms_single_face')
        if quality.get('status')!='pilot_basic_checks_passed':return end('review','source_basic_usability_not_met')
        n=obs['nisp'];s=n['scores'];m=r.data['measurements']
        if n['status']!='success' or any(type(s[k]) not in (float,int) or not math.isfinite(s[k]) or not 0<=s[k]<=1 for k in ['mouth','blink','frontal','left','right']):
            return end('error','nisp_unavailable_or_invalid')
    except (KeyError,ValueError,TypeError):return end('error','required_source_observation_missing')
    yaw,pitch,roll=[m.get(k+'_degrees') for k in ['yaw','pitch','roll']]
    pose_available=all(type(v) in (float,int) and math.isfinite(v) for v in [yaw,pitch,roll])
    view=pose_available and abs(yaw)<=25 and abs(pitch)<=30 and abs(roll)<=20
    lip=m.get('lip_ratio');eyes=m.get('eye_ratios');actions=[]
    if view and lip is not None and lip<=config.closed_lip_max and s['mouth']<=config.mouth_closed_score_max:actions.append('mouth_closed_to_open')
    if view and lip is not None and lip>=config.open_lip_min and s['mouth']>=config.mouth_open_score_min:actions.append('mouth_open_to_closed')
    if (view and eyes is not None and len(eyes)==2 and min(eyes)>=config.eye_open_min
        and abs(eyes[0]-eyes[1])<=config.eye_asymmetry_max and s['blink']<=config.blink_score_max):actions.append('eyes_open_to_closed')
    frontal=pose_available and abs(yaw)<=config.frontal_yaw_max and abs(pitch)<=config.frontal_pitch_max and abs(roll)<=config.frontal_roll_max and max(s['left'],s['right'])<=config.side_score_max
    normal_pose=frontal and s['frontal']>=config.frontal_score_min
    fallback=not actions and frontal and s['frontal']>=config.fallback_frontal_score_min
    if normal_pose or fallback:actions+=['frontal_to_left','keep_frontal','frontal_to_right']
    r.data.update(candidate_actions=actions,fallback_pose=bool(fallback),action_decisions={a:dict(eligible=a in actions,reason='concordant_pilot_observations' if a in actions else 'source_not_suitable_or_unknown') for a in FAMILIES},
        state_semantics='pilot estimates; eye visibility proxy; no truth/probability or production calibration')
    return end('eligible' if actions else 'review','pose_fallback_pilot' if fallback else 'supported_actions_available' if actions else 'no_suitable_action_or_pose_fallback')


class Day10Scheduler:
    def __init__(self,seed=2026102901):
        self.seed=seed;self.families=Counter();self.actions=Counter();self.levels=Counter();self.seen=set();self.last_selection=None
    def select(self,decision):
        d=decision.data;source=d['source_sha256'];actions=d.get('candidate_actions',[])
        if d['eligibility']!='eligible' or not actions or source in self.seen:return None
        tie=lambda s:hashlib.sha256((str(self.seed)+':'+source+':'+s).encode()).hexdigest()
        families={FAMILIES[a] for a in actions}
        family=min(families,key=lambda f:(self.families[f],tie(f)))
        action=min([a for a in actions if FAMILIES[a]==family],key=lambda a:(self.actions[a],tie(a)))
        levels=['canonical'] if action in ('mouth_open_to_closed','eyes_open_to_closed','keep_frontal') else ['L1','L2','L3']
        level=min(levels,key=lambda l:(self.levels[action,l],tie(action+':'+l)))
        self.last_selection=dict(seed=self.seed,family_counts_before=dict(self.families),action_counts_before=dict(self.actions),candidates=actions,selected_family=family,selected_action=action,selected_level=level)
        self.families[family]+=1;self.actions[action]+=1;self.levels[action,level]+=1;self.seen.add(source)
        return select_day10_profile(action,level)
