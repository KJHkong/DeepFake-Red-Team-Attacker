"""Deterministic metadata filters, exact SHA de-duplication and explainable Top-K."""
from time import perf_counter

from shared.schemas import ErrorInfo, ReferenceResult, Status


class MetadataRetriever:
    def __init__(self,snapshot):self.snapshot=snapshot

    def search(self,query):
        started=perf_counter();info=self.snapshot.info
        common={'query':query.model_copy(deep=True),'manifest_version':info.manifest_version,
                'manifest_sha256':info.manifest_sha256}
        if query.split!='retrieval_reference' or (query.manifest_version and query.manifest_version!=info.manifest_version):
            code='split_not_retrievable' if query.split!='retrieval_reference' else 'manifest_version_mismatch'
            return ReferenceResult(**common,status=Status.INVALID_INPUT,elapsed_ms=(perf_counter()-started)*1000,
                empty_reason=code,error=ErrorInfo(code=code,message='Query does not match the permitted retrieval snapshot'))
        records=self.select(query)
        references=[]
        for rank,record in enumerate(records[:query.limit],1):
            record.rank=rank;references.append(record)
        return ReferenceResult(**common,references=references,matched_count=len(records),
            elapsed_ms=(perf_counter()-started)*1000,
            empty_reason='no_matching_authorized_references' if not records else None)

    def select(self,query):
        """All filtered, SHA-deduplicated candidates, without premature Top-K."""
        from shared.contracts import ModuleFailure
        if query.split!='retrieval_reference':
            raise ModuleFailure('split_not_retrievable','Only retrieval-reference candidates are permitted')
        if query.manifest_version and query.manifest_version!=self.snapshot.info.manifest_version:
            raise ModuleFailure('manifest_version_mismatch','Query version differs from the active snapshot')
        excluded_sha=set(query.exclude_sha256)
        excluded_sha.update(r.artifact.sha256 for r in self.snapshot.records if r.sample_id in query.exclude_sample_ids)
        excluded_sha.update(r.artifact.sha256 for r in self.snapshot.records if r.lineage_group in query.exclude_lineage_groups)
        filtered=[]
        for record in self.snapshot.eligible:
            if record.attack_scope in query.exclude_attack_scopes:continue
            if record.sample_id in query.exclude_sample_ids or record.artifact.sha256 in excluded_sha:continue
            if record.lineage_group and record.lineage_group in query.exclude_lineage_groups:continue
            if set(query.required_permitted_use)-set(record.permitted_use):continue
            if query.require_display_allowed and (not record.artifact.display_allowed or 'display' not in record.permitted_use):continue
            matches={'permission_status':'authorized','validation_status':'verified','split':'retrieval_reference'}
            keep=True
            for field in ['truth_label','directory_label','collection','capture_method','attack_scope','review_status',
                          'reference_role','attack_family','generation_method','domain','device','scene']:
                expected=getattr(query,field)
                if expected is not None:
                    if getattr(record,field)!=expected:keep=False;break
                    matches[field]=expected
            if not keep:continue
            for dimension in ['width','height']:
                actual=getattr(record.artifact,dimension)
                low,high=getattr(query,'min_'+dimension),getattr(query,'max_'+dimension)
                if (low is not None and (actual is None or actual<low)) or (high is not None and (actual is None or actual>high)):
                    keep=False;break
                if low is not None:matches['min_'+dimension]=low
                if high is not None:matches['max_'+dimension]=high
            if keep:
                if query.exclude_attack_scopes:matches['exclude_attack_scopes']=sorted(set(query.exclude_attack_scopes))
                matches['required_permitted_use']=sorted(set(query.required_permitted_use)|{'metadata_retrieval'})
                if query.require_display_allowed:matches['display_allowed']=True
                filtered.append((record,matches))
        filtered.sort(key=lambda item:(item[0].priority,item[0].sample_id))
        unique={}
        for record,matches in filtered:
            sha=record.artifact.sha256
            if sha in unique:
                unique[sha].duplicate_sample_ids.append(record.sample_id)
                continue
            unique[sha]=record.model_copy(update={'duplicate_sample_ids':[], 'matched_conditions':matches,
                'rank':None,'similarity_score':None,'similarity_metric':None,
                'ranking_reason':f'priority={record.priority}; sample_id ascending (no visual similarity score)'},deep=True)
        records=list(unique.values())
        for record in records:record.duplicate_sample_ids.sort()
        return records
