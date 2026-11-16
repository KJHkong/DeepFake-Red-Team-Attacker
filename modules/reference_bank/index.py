"""Read-only versioned bank. No legacy auto-import, recursive scan, embedding or Gate."""
from copy import deepcopy
from time import perf_counter

from shared.contracts import ArtifactAccess, ModuleFailure
from shared.schemas import ErrorInfo, ModuleResult, ReferenceQuery, ReferenceRecord, Status, now
from modules.reference_bank.manifest import ReferenceFiles, load_manifest
from modules.reference_bank.retrieval import MetadataRetriever


class EmptyReferenceBank:
    def retrieve(self, query: ReferenceQuery) -> list[ReferenceRecord]:
        return []


class ManifestReferenceBank:
    def __init__(self, bank_root, *, image_access: ArtifactAccess | None = None, manifest_name=None):
        self._manifest_name=manifest_name
        self._snapshot=load_manifest(bank_root,manifest_name=manifest_name)
        self._retriever=MetadataRetriever(self._snapshot)
        self._files=ReferenceFiles(bank_root)
        self._image_access=image_access

    @property
    def audit(self):return deepcopy(self._snapshot.audit)

    @property
    def manifest_info(self):return self._snapshot.info.model_copy(deep=True)

    def assert_current(self):
        """Visual cache use requires the current admission snapshot, unlike metadata-only search."""
        current=load_manifest(self._files.root,manifest_name=self._manifest_name)
        if current.info!=self._snapshot.info:
            raise ModuleFailure('manifest_version_mismatch','Active admission metadata changed; load a new Bank and index')

    def select(self,query):return self._retriever.select(query)

    def embedding_source(self,sample_id):
        """Authorize and resolve metadata only; pixel access remains a separate explicit call."""
        records=[r for r in self._snapshot.records if r.sample_id==sample_id]
        if len(records)!=1:raise ModuleFailure('reference_not_found','Embedding reference identity is absent or ambiguous')
        record=records[0]
        if (record.permission_status!='authorized' or record.validation_status!='verified' or
                not {'image_read','embedding'}<=set(record.permitted_use) or
                record.split not in {'retrieval_reference','development'} or self._image_access is None):
            raise ModuleFailure('permission_denied','Embedding requires explicit purposes, verified source and a server image gateway')
        path=self._files.inspect_image(record.artifact.path,size_bytes=record.artifact.size_bytes)
        artifact=record.artifact.model_copy(update={'path':path,
            'display_allowed':record.artifact.display_allowed and 'display' in record.permitted_use},deep=True)
        return artifact

    def read_for_embedding(self,sample_id):
        self.assert_current()
        artifact=self.embedding_source(sample_id)
        return artifact,self._image_access.read(artifact)

    def search(self,query):return self._retriever.search(query)

    def retrieve(self,query: ReferenceQuery) -> list[ReferenceRecord]:
        """Existing internal list port, backed by the same retrieval implementation."""
        result=self.search(query)
        if result.error:raise ModuleFailure(result.error.code,result.error.message)
        return result.references

    def verify_sample(self,sample_id):
        """Explicit authorized image verification, separate from metadata loading/search.

        It never edits the manifest or upgrades pending records automatically.
        """
        started,stamp=perf_counter(),now();pixels=False;mock=False
        try:
            records=[r for r in self._snapshot.records if r.sample_id==sample_id]
            if len(records)!=1:raise ModuleFailure('reference_not_found','Reference is absent or has ambiguous identity')
            record=records[0];mock=record.artifact.mock
            if record.permission_status!='authorized' or 'image_read' not in record.permitted_use:
                raise ModuleFailure('permission_denied','Reference has no explicit image-read permission')
            if self._image_access is None:raise ModuleFailure('permission_denied','No server image-access policy was injected')
            path=self._files.inspect_image(record.artifact.path,size_bytes=record.artifact.size_bytes)
            artifact=record.artifact.model_copy(update={'path':path},deep=True)
            pixels=True
            self._image_access.read(artifact)
            return ModuleResult(module='reference_bank_validation',status=Status.SUCCESS,mock=artifact.mock,
                started_at=stamp,elapsed_ms=(perf_counter()-started)*1000,inputs=[artifact.artifact_id],
                data={'sample_id':sample_id,'image_pixels_read':True,'manifest_updated':False,'sha256':artifact.sha256})
        except ModuleFailure as exc:
            return ModuleResult(module='reference_bank_validation',status=Status.MODULE_FAILED,mock=mock,
                started_at=stamp,elapsed_ms=(perf_counter()-started)*1000,
                data={'sample_id':sample_id,'image_pixels_read':pixels,'manifest_updated':False},
                error=ErrorInfo(code=exc.code,message=str(exc)))
