"""Immutable, opt-in Day9 controls; historical and default profiles are untouched."""
import hashlib
import json
from pathlib import Path

BANK = Path(__file__).resolve().parents[3]/'configs/liveportrait_day9_actions_v1.json'
BANK_SHA = '046a8a7a6fb5cab5df532ed661b1fc5fb6cb31e999bc71ce8b261247500c106d'
VERSION = 'liveportrait_action_tuning_v1_1028'


def entries():
    raw = BANK.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BANK_SHA:
        raise ValueError('Day9 frozen controls changed')
    return json.loads(raw)['entries']


def resolve(prompt_id, prompt):
    entry = next((r for r in entries() if r['prompt_id'] == prompt_id), None)
    if entry is None or entry['prompt_text'] != prompt:
        raise ValueError('Unknown Day9 action/Prompt pair')
    return entry


