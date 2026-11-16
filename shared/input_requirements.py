"""Shared role/prompt contract checks; no filesystem access or execution."""
from collections import Counter

from shared.contracts import ModuleFailure
from shared.schemas import AttackRequest, ToolManifest


def validate_requirements(request: AttackRequest, manifest: ToolManifest | None):
    if manifest is None:
        if not request.inputs and not (request.prompt and request.prompt.strip()):
            raise ModuleFailure("missing_input", "Provide image inputs or a nonempty prompt", field="inputs")
        return
    if manifest.kind != "generator":
        raise ModuleFailure("invalid_input", "The generator field must select a Generator manifest", field="generator")
    if request.mode == "real" and manifest.mock:
        raise ModuleFailure("mode_tool_mismatch", "A mock tool cannot execute in real mode", field="generator")
    if request.mode == "mock" and not manifest.mock:
        raise ModuleFailure("mode_tool_mismatch", "Mock mode requires an explicit mock tool", field="generator")
    if request.taxonomy not in manifest.taxonomy:
        raise ModuleFailure("taxonomy_mismatch", "Taxonomy is not supported by the selected contract", field="taxonomy")
    requirements = manifest.input_requirements
    allowed = {r.role: r for r in requirements.roles}
    counts = Counter(image.role for image in request.inputs)
    for index, image in enumerate(request.inputs):
        if image.role not in allowed:
            raise ModuleFailure("input_role_mismatch", "Image role is not accepted by the selected tool; use validate_inputs for input-only checks",
                                field=f"inputs.{index}.role")
    for role, requirement in allowed.items():
        if counts[role] < requirement.min_count:
            raise ModuleFailure("missing_input", f"Required role '{role}' needs at least {requirement.min_count} image(s)", field="inputs")
        if counts[role] > requirement.max_count:
            raise ModuleFailure("input_role_mismatch", f"Role '{role}' permits at most {requirement.max_count} image(s)", field="inputs")
    if requirements.prompt_required and not (request.prompt and request.prompt.strip()):
        raise ModuleFailure("missing_prompt", "The selected tool requires a nonempty prompt", field="prompt")
    if request.mode != "mock" and request.prompt and request.prompt.strip() and not (requirements.prompt_required or requirements.accepts_prompt):
        raise ModuleFailure("capability_mismatch", "The selected tool does not consume text prompts", field="prompt")

