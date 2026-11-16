"""Exact frozen text/action -> bounded official LivePortrait numeric controls.

This adapter is not a text encoder. Unknown/free-form prompts are unsupported.
"""
import hashlib
import json
from pathlib import Path

PROMPTS=Path(__file__).resolve().parents[3]/'configs/prompts/v2_edit_50_actions_v1_1023.yaml'
PROMPT_CONFIG_SHA='6347d240be627d23fcf478165a15494e729a9153e2e0bd8081ff03b19b6224c6'


def controls_for(action_id,prompt):
    raw=PROMPTS.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROMPT_CONFIG_SHA:raise ValueError('Frozen Prompt config changed')
    action=next((a for a in json.loads(raw)['actions'] if a['action_id']==action_id),None)
    if action is None or prompt!=action['complete_prompt']:raise ValueError('Unsupported action/Prompt pair')
    target=action['target_state']
    return dict(action_id=action_id,adapter_version='canonical_static_controls_v1_1023',
        native_text_conditioning=False,full_prompt_retained=True,
        lip_ratio=0.0 if target=='mouth_closed' else .3 if target=='mouth_open' else None,
        eye_ratio=0.0 if target=='eyes_closed' else .6 if target=='eyes_open' else None,
        target_yaw_degrees={'head_left':20.0,'head_right':-20.0,'head_frontal':0.0}.get(target),
        pose_coordinate_convention='official motion yaw +20 = viewer left / -20 = viewer right; to be visually checked in Pilot',
        no_driving_asset=True,preservation='keep source pitch/roll/expression except requested control; official stitching and paste-back')
