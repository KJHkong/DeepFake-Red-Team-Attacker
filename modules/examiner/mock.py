from PIL import Image

from shared.schemas import AttackRequest, Candidate, ExaminerResult, ReferenceRecord


class MockExaminer:
    def examine(self, request: AttackRequest, candidate: Candidate, references: list[ReferenceRecord]) -> ExaminerResult:
        with Image.open(candidate.artifact.path) as image:
            image.load()
            size_ok = image.width >= request.constraints.min_width and image.height >= request.constraints.min_height
        scenario = request.mock_scenario
        injected_failure = scenario in {"gate1_reject", "refine_reject", "no_face"} or (
            scenario == "refine_success" and candidate.revision == 0)
        passed = size_ok and not injected_failure
        reasons = [] if passed else ["mock_no_face" if scenario == "no_face" else "mock_gate1_failure_or_size_constraint"]
        return ExaminerResult(candidate_id=candidate.candidate_id, validity_passed=passed,
                              constraints_passed=size_ok, distribution_check="passed" if passed else "failed",
                              gate1_passed=passed, face_present=None, reasons=reasons, mock=True,
                              reference_ids=[r.sample_id for r in references])
