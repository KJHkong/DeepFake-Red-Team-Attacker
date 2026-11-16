"""Run Day 1 fixtures without external services or business media."""
import argparse
import json
from pathlib import Path

from core.orchestrator import Orchestrator
from shared.schemas import AttackRequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--topic", default="overall_interface", help="Readable output topic, e.g. generator_inswapper")
    args = parser.parse_args()
    scenarios = ["accepted", "gate1_reject", "gate2_reject", "refine_success", "refine_reject", "generation_failed", "detector_failed", "no_face"] if args.all else ["accepted", "gate2_reject"]
    orchestrator = Orchestrator(args.runs_root, output_topic=args.topic)
    for scenario in scenarios:
        run = orchestrator.run(AttackRequest(mock_scenario=scenario))
        assert run.verification_sheet is not None
        print(json.dumps({"scenario": scenario, "run_id": run.run_id, "state": run.state,
                          "mock": run.mock, "sheet": run.verification_sheet.image_path}))


if __name__ == "__main__":
    main()
