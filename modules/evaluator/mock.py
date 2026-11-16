from time import perf_counter
from uuid import uuid4

from shared.schemas import (AttackRequest, Candidate, DetectorConfig, DetectorResult,
                            ErrorInfo, EvaluationResult, Status)


class MockEvaluator:
    def evaluate(self, request: AttackRequest, candidate: Candidate) -> EvaluationResult:
        started = perf_counter()
        config = DetectorConfig()
        failed = request.mock_scenario == "detector_failed"
        score = 0.8 if request.mock_scenario == "gate2_reject" else 0.1
        detector = DetectorResult(
            artifact_id=candidate.artifact.artifact_id, request_uuid=uuid4().hex,
            status=Status.DETECTOR_FAILED if failed else Status.SUCCESS, mock=True,
            fake_score=None if failed else score, real_score=None if failed else 1-score,
            predicted_fake=None if failed else score >= config.threshold,
            threshold=config.threshold, service=config, score_type=config.score_type,
            error=ErrorInfo(code="detector_failed", message="Injected mock Detector failure") if failed else None,
        )
        success = None if failed else not detector.predicted_fake
        return EvaluationResult(candidate_id=candidate.candidate_id, status=detector.status,
                                detector=detector, attack_success=success, gate2_passed=success,
                                mock=True, cost_ms=(perf_counter()-started)*1000,
                                reasons=["mock_only; quality and identity metrics not evaluated"])
