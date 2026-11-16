"""Bounded, no-link metadata loading. Inspect file metadata only; never decode images."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import re

from pydantic import ValidationError

from shared.contracts import ModuleFailure
from shared.schemas import ReferenceManifest, ReferenceRecord


class ReferenceFiles:
    def __init__(self, root):
        self.root=Path(root).absolute()
        if '..' in self.root.parts or self.root==Path('/'):
            raise ValueError('Reference root must be a scoped server directory')

    def _path(self, relative, *, image=False):
        if not isinstance(relative,str) or '\0' in relative:
            raise ModuleFailure('path_not_allowed','Invalid reference path')
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts or (image and path.parts[0]!='data'):
            raise ModuleFailure('path_not_allowed','Reference path must remain inside the permitted data directory')
        return self.root/path

    def _open(self, relative, *, image=False):
        path=self._path(relative,image=image)
        fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        try:
            for part in path.parts[1:-1]:
                nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=fd)
                os.close(fd);fd=nxt
            result=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=fd)
            info=os.fstat(result)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:
                os.close(result)
                raise ModuleFailure('path_not_allowed','Reference must be a standalone regular file')
            return path,result,info
        except FileNotFoundError:
            raise ModuleFailure('file_not_found','Reference metadata or sample is missing') from None
        except OSError:
            raise ModuleFailure('path_not_allowed','Reference path is inaccessible or contains a link') from None
        finally:os.close(fd)

    def read_metadata(self, relative, *, max_bytes):
        _,fd,before=self._open(relative)
        with os.fdopen(fd,'rb') as stream:
            if before.st_size>max_bytes:
                raise ModuleFailure('manifest_too_large','Reference metadata exceeds its byte limit')
            raw=stream.read(max_bytes+1);after=os.fstat(stream.fileno())
        if len(raw)>max_bytes:raise ModuleFailure('manifest_too_large','Reference metadata exceeds its byte limit')
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
            raise ModuleFailure('input_changed','Reference metadata changed during the bounded read')
        return raw

    def inspect_image(self, relative, *, size_bytes=None):
        path,fd,info=self._open(relative,image=True)
        os.close(fd)  # No image bytes are read.
        if info.st_size<=0 or info.st_size>20*1024**2:
            raise ModuleFailure('file_too_large','Reference sample has an invalid byte size')
        if size_bytes is not None and size_bytes!=info.st_size:
            raise ModuleFailure('invalid_artifact','Reference size differs from its manifest')
        return str(path)


@dataclass(frozen=True)
class ManifestSnapshot:
    info: ReferenceManifest
    records: tuple[ReferenceRecord,...]
    eligible: tuple[ReferenceRecord,...]
    audit: dict


def load_manifest(root, *, manifest_name=None):
    files=ReferenceFiles(root)
    if manifest_name is not None and (not isinstance(manifest_name,str) or
            re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}',manifest_name) is None):
        raise ModuleFailure('path_not_allowed','Named Manifest must be an opaque server-owned identifier')
    directory='manifests' if manifest_name is None else 'manifests/'+manifest_name
    try:
        info=ReferenceManifest.model_validate_json(files.read_metadata(directory+'/reference_index.json',max_bytes=16*1024))
    except ValidationError:
        raise ModuleFailure('invalid_manifest','Reference index header violates its contract') from None
    if manifest_name is not None and info.manifest_version!=manifest_name:
        raise ModuleFailure('manifest_version_mismatch','Selected Manifest name differs from its declared version')
    raw=files.read_metadata(directory+'/reference_active.jsonl',max_bytes=8*1024**2)
    if hashlib.sha256(raw).hexdigest()!=info.manifest_sha256:
        raise ModuleFailure('manifest_digest_mismatch','Active manifest does not match its declared snapshot')
    lines=raw.splitlines()
    if len(lines)!=info.record_count:
        raise ModuleFailure('manifest_count_mismatch','Active manifest row count differs from the index header')
    rows=[];issues=[];excluded=Counter()
    for index,line in enumerate(lines,1):
        try:
            if len(line)>64*1024:raise ValueError('bounded line')
            entry=json.loads(line)
        except (ValueError,RecursionError):
            issues.append({'line':index,'code':'invalid_json'});continue
        try:
            # Artifact identity is mandatory in a versioned manifest, rather than generated on every load.
            if not isinstance(entry,dict) or 'artifact_id' not in entry.get('artifact',{}):
                raise ValueError('stable Artifact identity required')
            record=ReferenceRecord.model_validate(entry)
        except (ValidationError,ValueError,TypeError):
            issues.append({'line':index,'code':'invalid_record'});continue
        if record.manifest_version!=info.manifest_version:
            raise ModuleFailure('manifest_version_mismatch','Record version differs from its index header')
        rows.append((index,record))
    counts=Counter(r.sample_id for _,r in rows)
    duplicate_ids={key for key,count in counts.items() if count>1}
    artifact_counts=Counter(r.artifact.artifact_id for _,r in rows)
    duplicate_artifact_ids={key for key,count in artifact_counts.items() if count>1}
    holdout_paths={r.artifact.path for _,r in rows if r.split=='holdout'}
    groups=defaultdict(list)
    for _,record in rows:
        if record.artifact.sha256:groups[record.artifact.sha256].append(record)
    holdout={sha for sha,group in groups.items() if any(r.split=='holdout' for r in group)}
    conflicts={sha for sha,group in groups.items() if len({r.truth_label for r in group if r.truth_label!='unknown'})>1}
    eligible=[]
    for index,record in rows:
        reason=None
        if record.sample_id in duplicate_ids:
            reason='duplicate_sample_id';issues.append({'line':index,'code':reason})
        elif record.artifact.artifact_id in duplicate_artifact_ids:
            reason='duplicate_artifact_id';issues.append({'line':index,'code':reason})
        elif record.permission_status!='authorized' or 'metadata_retrieval' not in record.permitted_use:
            reason='permission_denied'
        elif record.validation_status!='verified':reason='not_verified'
        elif record.split!='retrieval_reference':reason='split_not_retrievable'
        elif record.artifact.sha256 in holdout or record.artifact.path in holdout_paths:reason='holdout_sha_overlap'
        elif record.artifact.sha256 in conflicts:reason='conflicting_truth_labels'
        if reason:
            excluded[reason]+=1;continue
        try:
            absolute=files.inspect_image(record.artifact.path,size_bytes=record.artifact.size_bytes)
            artifact=record.artifact.model_copy(update={'path':absolute,
                'display_allowed':record.artifact.display_allowed and 'display' in record.permitted_use},deep=True)
            eligible.append(record.model_copy(update={'artifact':artifact},deep=True))
        except ModuleFailure as exc:
            excluded[exc.code]+=1;issues.append({'line':index,'code':exc.code})
    audit={'manifest_version':info.manifest_version,'manifest_sha256':info.manifest_sha256,
        'declared_rows':info.record_count,'parsed_rows':len(rows),'eligible_rows_before_sha_dedup':len(eligible),
        'eligible_unique_sha256':len({r.artifact.sha256 for r in eligible}),
        'duplicate_sample_ids':sorted(duplicate_ids),'excluded_by_reason':dict(excluded),'issues':issues,
        'image_pixels_read':False,'integrity_scope':'Header/records and path/file metadata checked; image SHA/decode retained from explicit verification evidence, not re-read'}
    return ManifestSnapshot(info,tuple(r for _,r in rows),tuple(eligible),audit)
