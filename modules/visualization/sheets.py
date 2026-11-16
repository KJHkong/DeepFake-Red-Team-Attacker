"""The same sheet format is used by module local evals and complete runs."""
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

from shared.output_paths import dated_name
from shared.schemas import RunMetadata, SheetPanel, VerificationSheet


def render_sheet(spec: VerificationSheet, *, allow_business_display: bool = False, columns: int = 4) -> VerificationSheet:
    if type(columns) is not int or not 1 <= columns <= 6:
        raise ValueError("Sheet columns must be an integer from 1 to 6")
    cell_width, cell_height = 300, 300
    rows = (len(spec.panels) + columns - 1) // columns
    image = Image.new("RGB", (columns * cell_width, 110 + rows * cell_height + 90), (21, 28, 41))
    draw = ImageDraw.Draw(image)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).is_file() else ImageFont.load_default()
    title = ImageFont.truetype(font_path, 23) if Path(font_path).is_file() else font
    draw.text((18, 14), f"Verification Sheet  |  mock={str(spec.mock).lower()}", font=title, fill=(100, 212, 238))
    draw.text((18, 49), f"Run ID: {spec.run_id}", font=font, fill="white")
    draw.text((18, 73), "Originals and stage results are retained in run metadata.", font=font, fill=(170, 182, 198))
    for index, panel in enumerate(spec.panels):
        x, y = (index % columns) * cell_width, 110 + (index // columns) * cell_height
        draw.rounded_rectangle((x+8, y+5, x+292, y+292), radius=10, fill=(33, 43, 59))
        draw.text((x+18, y+15), panel.label, font=font, fill="white")
        artifact = panel.artifact
        allowed = artifact is not None and artifact.display_allowed and (artifact.mock or allow_business_display)
        note = panel.note
        if allowed:
            with Image.open(artifact.path) as source:
                preview = ImageOps.contain(source.convert("RGB"), (260, 220))
            image.paste(preview, (x+20+(260-preview.width)//2, y+43+(220-preview.height)//2))
        elif artifact is None and panel.text:
            lines = [line for text in panel.text for line in textwrap.wrap(text, width=29)]
            for row, line in enumerate(lines[:10]):
                draw.text((x+18, y+48+row*20), line, font=font, fill=(210, 221, 235))
        else:
            note = "Withheld: display permission absent" if artifact else panel.note
            draw.text((x+24, y+125), "WITHHELD" if artifact else "N/A", font=title, fill=(158, 173, 196))
        draw.text((x+18, y+268), textwrap.shorten(note, width=35, placeholder="..."), font=font, fill=(180, 195, 211))
    footer_y = 116 + rows*cell_height
    for line in spec.summary[:3]:
        draw.text((18, footer_y), line[:142], font=font, fill="white")
        footer_y += 23
    path = Path(spec.image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    Path(spec.spec_path).write_text(spec.model_dump_json(indent=2)+"\n")
    return spec


def task_input_panels(run, inputs):
    """Use actual/requested roles; zero-image tasks get configuration provenance, never invented parents."""
    panels = []
    required_roles = list(dict.fromkeys([asset.role for asset in run.request.inputs]))
    labels = {"source":"Source", "target":"Target", "image":"Input Image", "reference":"Reference", "driving":"Driving"}
    for role in required_roles:
        matches = [a for a in inputs if a.role == role]
        if not matches:
            panels.append(SheetPanel(label=labels[role], note="N/A: input snapshot not available"))
        for index, artifact in enumerate(matches, 1):
            label = labels[role] + (f" #{index}" if len(matches)>1 else "")
            panels.append(SheetPanel(label=label, artifact=artifact,
                note=f"{artifact.width}x{artifact.height} {artifact.media_type} / input"))
    if run.request.prompt is not None:
        panels.append(SheetPanel(label="Instruction Prompt", text=[run.request.prompt],
            note="Full instruction retained in metadata"))
    if not required_roles:
        parameters = run.plan.parameters if run.plan else run.request.parameters
        panels.append(SheetPanel(label="Seed / Sampling Parameters", text=[
            f"seed={run.request.seed}", *[f"{k}={v}" for k,v in parameters.items() if k!='seed'],
            "Image input=N/A"], note="Task configuration; no input Artifact"))
        generator = run.candidates[0] if run.candidates else None
        provenance = generator.metadata.get("generation_task", {}) if generator else {}
        checkpoint = provenance.get("checkpoint_info", {})
        model_id = checkpoint.get("model_id", "N/A: generation not completed")
        runtime = generator.metadata.get("runtime", {}) if generator else {}
        panels.append(SheetPanel(label="Model / Checkpoint", text=[
            str(model_id), f"version={provenance.get('tool_version', 'N/A')}",
            f"training={runtime.get('training', 'not_evaluated')}",
            str(runtime.get("checkpoint_scope", "Exact checkpoint and SHA-256 in metadata"))],
            note="Model provenance; no parent image"))
    return panels


def content_panels(run, inputs, generated):
    """Use recorded step results, including partial failures; never manufacture stage images."""
    def picture(label, artifact, status="success", note=None):
        description = (f"{artifact.width}x{artifact.height} {artifact.media_type} / {status}" if artifact else f"N/A: {status}")
        return SheetPanel(label=label, artifact=artifact, note=note or description)
    panels = task_input_panels(run, inputs)
    panels.append(picture("Generator Output", generated, "success" if generated else "not_run / failed"))
    stage = next((s for s in run.stages if s.module == "modifier"), None)
    invocation = stage.data.get("modifier_invocation", {}) if stage else {}
    results = invocation.get("data", {}).get("step_results", [])
    artifacts = {a.artifact_id:a for a in run.artifacts}
    operations = run.plan.modifier_chain if run.plan else []
    for index in range(max(2, len(operations))):
        result = results[index] if index < len(results) else None
        operation = operations[index] if index < len(operations) else None
        artifact_id = (result.get("candidate") or {}).get("artifact", {}).get("artifact_id") if result else None
        status = result['status'] if result else "not_executed" if operation else "not_planned"
        name = operation.tool_name if operation else "N/A"
        panels.append(picture(f"Modifier Step {index+1}: {name}", artifacts.get(artifact_id), status))
    final = next((c.artifact for c in run.candidates if c.candidate_id == run.final_candidate_id), None)
    panels.append(picture("Final Candidate", final, run.state,
        "Unmodified Generator reference" if final and not operations else None))
    generation = next((s for s in run.stages if s.module == "generator"), None)
    panels.append(SheetPanel(label="Generator / Execution", text=[
        f"tool={run.plan.tool_name if run.plan else None}",
        f"request seed={run.request.seed}", f"status={generation.status if generation else 'not_run'}",
        f"elapsed_ms={generation.elapsed_ms:.2f}" if generation else "elapsed_ms=N/A",
        f"scope={run.request.execution_scope}", f"state={run.state}",
        *[f"{k}={v}" for k,v in (run.plan.parameters if run.plan else {}).items()]], note="Real content; Gates not evaluated"))
    generator = run.candidates[0] if run.candidates else None
    preprocess = generator.metadata.get("runtime", {}).get("preprocess") if generator else None
    if preprocess:
        panels.append(SheetPanel(label="Actual Preprocessing", text=[str(preprocess)], note="Worker-recorded; no identity claim"))
    for index, operation in enumerate(operations):
        result = results[index] if index < len(results) else None
        parameters = result['parameters'] if result else operation.parameters
        panels.append(SheetPanel(label=f"Step {index+1} / Parameters", text=[
            operation.tool_name, *[f"{key}={value}" for key,value in parameters.items()],
            f"status={result['status'] if result else 'not_run'}",
            f"elapsed_ms={result['elapsed_ms']:.2f}" if result else "elapsed_ms=N/A",
            f"step_id={run.plan.steps[index+1].step_id}"], note="Actual parameters" if result else "Planned; not executed"))
    panels.extend([SheetPanel(label="Business Reference", note="N/A: not executed"),
                   SheetPanel(label="Refiner Output", note="N/A: not executed"),
                   SheetPanel(label="Detector / Gate 1 / Gate 2", note="N/A: not evaluated")])
    return panels


def pipeline_sheet(run: RunMetadata, directory: Path, *, topic: str = "overall_interface") -> VerificationSheet:
    by_stage = {artifact.stage: artifact for artifact in run.artifacts}
    input_ids = set(run.input_artifact_ids)
    inputs = [artifact for artifact in run.artifacts if artifact.artifact_id in input_ids]
    by_role = {role: next((a for a in inputs if a.role == role), None) for role in ("source", "target")}
    final = run.candidates[-1].artifact if run.candidates else None
    def stage_note(name):
        artifact=by_stage.get(name)
        return f"mock={str(artifact.mock).lower()}" if artifact else "N/A: not executed"
    panels = [
        SheetPanel(label="Source", artifact=by_role["source"], note="Validated input snapshot" if by_role["source"] else "N/A: no source input"),
        SheetPanel(label="Target", artifact=by_role["target"], note="Validated input snapshot" if by_role["target"] else "N/A: no target input"),
        SheetPanel(label="Generator Output", artifact=by_stage.get("generator"), note=stage_note("generator")),
        SheetPanel(label="Modifier Output", artifact=by_stage.get("modifier"), note=stage_note("modifier")),
        SheetPanel(label="Business Reference", note="N/A: no business retrieval"),
        SheetPanel(label="Refiner Output", artifact=by_stage.get("refiner"), note=stage_note("refiner")),
        SheetPanel(label="Final Candidate", artifact=final, note=f"mock={str(final.mock).lower()}" if final else "N/A: no candidate generated"),
        SheetPanel(label="Prompt", text=[run.request.prompt] if run.request.prompt is not None else [],
            note="N/A" if run.request.prompt is None else "Recorded; not used by mock" if run.mock else "Instruction retained in metadata"),
    ]
    displayed_ids = {a.artifact_id for a in by_role.values() if a}
    panels.extend(SheetPanel(label=f"Input {artifact.role} #{index}", artifact=artifact, note="Validated input snapshot")
                  for index, artifact in enumerate(inputs, 1) if artifact.artifact_id not in displayed_ids)
    score = run.evaluation.detector.fake_score if run.evaluation else None
    score_type = run.evaluation.detector.score_type if run.evaluation else "N/A"
    summary = [f"state={run.state}   Gate1={run.gate1}   Gate2={run.gate2}   {score_type}_score={score}",
               "MOCK WORKFLOW ONLY: face/distribution/quality/identity are not real validations." if run.mock else
               "PLAN / INPUT PREPARATION ONLY: Generator, Gate1/Gate2 and quality evaluation were not executed.",
               "errors=" + ",".join(error.code for error in run.errors)]
    if run.purpose=="pipeline_smoke":
        summary[1]="REAL PIPELINE SMOKE: Generator + validity +0826 Detector; formal Gates/quality/ASR unassessed."
        panels.append(SheetPanel(label="0826 Detector",note="Real RPC; formal attack success N/A",text=[
            f"score_type={score_type}",f"fake_score={score}",
            f"threshold={run.evaluation.detector.threshold if run.evaluation else None}",
            f"predicted_fake={run.evaluation.detector.predicted_fake if run.evaluation else None}",
            "Gate1/Gate2/attack_success=N/A"]))
    if run.purpose == "generator_stage":
        summary[1] = "REAL GENERATOR STAGE ONLY: Examiner / Distribution / Detector / Gate1 / Gate2 / attack_success=N/A."
        generation = next((s for s in run.stages if s.module == "generator"), None)
        panels.append(SheetPanel(label="Generator Execution", note="Partial run; no final attack outcome", text=[
            f"tool={run.plan.tool_name if run.plan else None}",
            f"parameters={run.plan.parameters if run.plan else {}}",
            f"elapsed_ms={round(generation.elapsed_ms, 2) if generation else None}",
            f"status={generation.status if generation else 'not_run'}",
            "Downstream stages=not_evaluated"]))
    if run.purpose == "content_generation":
        panels = content_panels(run, inputs, by_stage.get("generator"))
        summary[1] = "REAL CONTENT GENERATION ONLY: Distribution / Quality / Identity / Refiner / Detector / Gate1 / Gate2=N/A."
    content = "content_only" if run.purpose == "content_generation" else "detector0826_smoke" if run.purpose=="pipeline_smoke" else run.request.mock_scenario if run.request.mode == "mock" else run.request.mode
    stem = dated_name(topic, f"{content}_verification", timestamp=run.started_at)
    return render_sheet(VerificationSheet(run_id=run.run_id, mock=run.mock, panels=panels,
                        image_path=str(directory / "visualizations" / f"{stem}.png"),
                        spec_path=str(directory / "visualizations" / f"{stem}.json"),
                        metadata_path=str(directory / "metadata.json"), summary=summary),
                        allow_business_display=run.request.constraints.allow_business_display)
