"""Core owns admission. Independent Examiner never certifies a final attack outcome."""
from shared.schemas import Gate1Decision, Status


def decide_gate1(run_id, candidate, result):
    reasons=[]
    if (result.assessment_scope!='per_image' or result.candidate_id!=candidate.candidate_id or
            result.artifact_id!=candidate.artifact.artifact_id or
            result.source_sha256!=candidate.artifact.sha256 or result.mock!=candidate.mock):
        decision='error';reasons=['examiner_candidate_binding_mismatch']
    elif result.execution_status==Status.INVALID_INPUT:
        decision='access_denied';reasons=['no_authorized_image_conclusion']
    elif result.execution_status!=Status.SUCCESS:
        decision='error';reasons=['examiner_execution_failed']
    elif any(c.required and c.status=='failed' for c in result.checks):
        decision='rejected';reasons=[c.name for c in result.checks if c.required and c.status=='failed']
    else:
        names=[c.name for c in result.checks]
        mandatory={'artifact_access','artifact_integrity','image_decode','image_safety','dimensions','task_semantics'}
        if (len(set(names))!=len(names) or not mandatory<=set(names) or
                any(c.required and c.status!='passed' for c in result.checks) or
                result.validity_passed is not True or result.constraints_passed is not True):
            decision='review';reasons=['required_evidence_incomplete']
        else:decision='passed'
    return Gate1Decision(run_id=run_id,candidate_id=candidate.candidate_id,
        artifact_id=candidate.artifact.artifact_id,source_sha256=candidate.artifact.sha256,
        decision=decision,reasons=reasons)
