"""Standard-library keep-frontal bank. All historical banks stay immutable."""
import hashlib
import json
from pathlib import Path

BANK=Path(__file__).resolve().parents[3]/'configs/liveportrait_keep_frontal_v1_1029.json'
VERSION='liveportrait_keep_frontal_v1_1029'


def resolve(prompt_id,prompt):
    bank=json.loads(BANK.read_text())
    if bank['bank_version']!=VERSION:raise ValueError('Unknown keep-frontal bank')
    entry=bank['entry']
    if entry['prompt_id']!=prompt_id or entry['prompt_text']!=prompt or entry['control_spec']!={'lip_ratio':None,'eye_ratio':None,'target_yaw_degrees':0.}:
        raise ValueError('Unexpected keep-frontal controls/Prompt')
    return entry


def bank_sha():return hashlib.sha256(BANK.read_bytes()).hexdigest()
