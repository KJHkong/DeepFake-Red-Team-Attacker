"""Standard-library frozen bank, preserving the accepted Day9 Worker binding."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from modules.generator.adapters.day9_actions import entries as parent_entries, VERSION as PARENT_VERSION, BANK_SHA as PARENT_SHA

ROOT=Path(__file__).resolve().parents[3]
BANK_PATH=ROOT/'configs/liveportrait_actions_frozen_v1_1028.json'
SCHEMA_PATH=ROOT/'configs/liveportrait_actions_frozen_v1_1028.schema.json'
VERSION='liveportrait_actions_frozen_v1_1028'
BANK_SHA='8720df8ec786bf440c49b02728ebd9eb085bf2ad8970c90410b029dc667f6c4f'
SCHEMA_SHA='e06e5255f021f7828c5e19b961291d49abb2148d3c1e7edbc1c28a595213079e'


def load_bank():
    raw=BANK_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BANK_SHA or hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()!=SCHEMA_SHA:
        raise ValueError('Frozen bank/Schema changed; publish a new version')
    bank=json.loads(raw)
    if bank['status']!='frozen' or bank['bank_version']!=VERSION or bank['entries']!=parent_entries():
        raise ValueError('Frozen settings differ from the accepted Day9 controls')
    return bank


def worker_binding(entry):
    """No new Worker/Registry or implicit admission: expose both provenance versions."""
    if entry not in load_bank()['entries']:raise ValueError('Unknown frozen configuration')
    return dict(frozen_config_version=VERSION,frozen_config_sha256=BANK_SHA,
        worker_parameters=dict(prompt_id=entry['prompt_id'],bank_version=PARENT_VERSION),
        worker_bank_sha256=PARENT_SHA,native_controls=deepcopy(entry['control_spec']),
        semantic_description=entry['prompt_text'],native_text_conditioning=False)
