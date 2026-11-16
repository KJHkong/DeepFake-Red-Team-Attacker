"""Independent immutable mouth profile. No model or API dependencies in Workers."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BANK=ROOT/'configs/liveportrait_mouth_bidirectional_v1_1028.json'
SCHEMA=ROOT/'configs/liveportrait_mouth_bidirectional_v1_1028.schema.json'
VERSION='liveportrait_mouth_bidirectional_v1_1028'
# Filled from the exact published configuration below; never infer values from text.
BANK_SHA='da5196adfed5fa14cca527d6e17764daaf44324ce30757b358850b04748cd263'
SCHEMA_SHA='b625891354744319db264b6bdefdfaec12e145665a607b952cbed03089408126'


def load_bank():
    data=BANK.read_bytes()
    if hashlib.sha256(data).hexdigest()!=BANK_SHA or hashlib.sha256(SCHEMA.read_bytes()).hexdigest()!=SCHEMA_SHA:
        raise ValueError('Mouth profile changed; publish a new version')
    bank=json.loads(data)
    parent=ROOT/bank['parent_config']['path']
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=bank['parent_config']['sha256']:
        raise ValueError('Parent frozen10 profile changed')
    return bank


def resolve(prompt_id,prompt):
    entry=next((e for e in load_bank()['entries'] if e['prompt_id']==prompt_id),None)
    if entry is None or entry['prompt_text']!=prompt:raise ValueError('Unknown mouth configuration/semantic text pair')
    return entry
