from pathlib import Path
from time import perf_counter

from shared.contracts import GeneratorTool, Modifier, ModuleFailure
from shared.schemas import (AttackRequest, Candidate, InputRequirements, RoleRequirement,
                            Taxonomy, ToolManifest, Artifact, ErrorInfo, ModuleResult, Plan, Status, now)


def pending_manifests() -> list[ToolManifest]:
    """Input contracts only: these declarations register no inference implementation."""
    common = dict(version="0.1", input_schema=AttackRequest.model_json_schema(),
                  output_schema=Candidate.model_json_schema(), backend="not_integrated", mock=False,
                  availability="not_integrated", capabilities=["input_contract_only"])
    return [
        ToolManifest(name="face_swap", taxonomy=[Taxonomy.FACE_SWAPPING], **common,
            input_requirements=InputRequirements(roles=[
                RoleRequirement(role="source", min_count=1), RoleRequirement(role="target", min_count=1),
                RoleRequirement(role="reference", max_count=6)])),
        ToolManifest(name="text_to_image", taxonomy=[Taxonomy.DIFFUSION], **common,
            input_requirements=InputRequirements(prompt_required=True)),
    ]


class GeneratorRegistry:
    def __init__(self):
        self._tools: dict[str, GeneratorTool] = {}
        self._manifests: dict[str, ToolManifest] = {}
        self._modifiers: dict[str, Modifier] = {}

    def register(self, tool: GeneratorTool):
        if tool.manifest.availability not in {"ready", "available"} or tool.manifest.kind != "generator":
            raise ValueError("Registered implementations must be ready")
        if tool.manifest.name in self._manifests:
            raise ValueError("Duplicate tool name")
        self._tools[tool.manifest.name] = tool
        self._manifests[tool.manifest.name] = tool.manifest

    def register_modifier(self, tool: Modifier):
        if tool.manifest.kind != "modifier" or tool.manifest.availability != "ready":
            raise ValueError("Modifier implementation must be ready and declared as modifier")
        if tool.manifest.name in self._manifests:
            raise ValueError("Duplicate tool name")
        self._modifiers[tool.manifest.name] = tool
        self._manifests[tool.manifest.name] = tool.manifest

    def is_available(self, name: str) -> bool:
        manifest = self._manifests.get(name)
        return bool(manifest and manifest.availability in {"ready", "available"} and
                    (name in self._tools if manifest.kind == "generator" else name in self._modifiers))

    def declare(self, manifest: ToolManifest):
        if manifest.availability not in {"not_integrated", "not_ready", "manifest_only", "disabled"} or manifest.name in self._manifests:
            raise ValueError("Expected a new unimplemented input contract")
        self._manifests[manifest.name] = manifest

    def manifest(self, name: str) -> ToolManifest:
        if name not in self._manifests:
            raise ModuleFailure("unsupported_generator", "Generator is unknown or not integrated", field="generator")
        return self._manifests[name]

    def get(self, name: str) -> GeneratorTool:
        if name not in self._tools or not self.is_available(name):
            raise ModuleFailure("unsupported_generator", "Generator is not integrated", field="generator")
        return self._tools[name]

    def get_modifier(self, name: str) -> Modifier:
        if name not in self._modifiers or not self.is_available(name):
            raise ModuleFailure("tool_unavailable", "Modifier is not available", field="modifier.tool_name")
        return self._modifiers[name]

    def manifests(self, *, taxonomy: Taxonomy | None = None) -> list[ToolManifest]:
        return [m for m in self._manifests.values() if taxonomy is None or taxonomy in m.taxonomy]

    def default_tool(self, taxonomy: Taxonomy) -> ToolManifest | None:
        tools = [m for m in self.manifests(taxonomy=taxonomy)
                 if m.kind == "generator" and not m.mock and self.is_available(m.name)]
        return min(tools, key=lambda m: (not m.is_default, m.default_priority if m.default_priority is not None else 10**9, m.name), default=None)

    def invoke(self, request: AttackRequest, plan: Plan, output_dir: Path, *,
               input_artifacts: list[Artifact]) -> ModuleResult:
        """Public execution boundary. The internal generate port may raise ModuleFailure."""
        from shared.input_requirements import validate_requirements
        started, stamp = perf_counter(), now()
        candidate, error, metadata = None, None, {}
        try:
            if plan.status != "ready" or not plan.tool_name:
                if plan.error:
                    raise ModuleFailure(plan.error.code, plan.error.message, field=plan.error.field)
                raise ModuleFailure("plan_failed", "An executable generator plan is required")
            tool = self.get(plan.tool_name)
            validate_requirements(request, tool.manifest)
            if (request.mode not in {"mock", "real"} or plan.execution_scope != request.execution_scope or plan.seed != request.seed or
                    plan.taxonomy != request.taxonomy or plan.mock != tool.manifest.mock or
                    plan.mock != (request.mode == "mock") or
                    (request.generator is not None and request.generator != plan.tool_name)):
                raise ModuleFailure("plan_failed", "Generator plan does not match the request")
            metadata = {"tool_name": tool.manifest.name, "version": tool.manifest.version,
                        "environment": tool.manifest.environment, "checkpoint_info": tool.manifest.checkpoint_info}
            candidate = tool.generate(request, plan, output_dir, input_artifacts=input_artifacts)
            if not isinstance(candidate, Candidate) or candidate.mock != tool.manifest.mock or candidate.artifact.mock != candidate.mock:
                candidate = None
                raise ModuleFailure("invalid_generator_output", "Generator returned an incompatible candidate")
        except ModuleFailure as exc:
            error = ErrorInfo(code=exc.code, message=str(exc), field=exc.field)
        except Exception:
            candidate = None
            error = ErrorInfo(code="generation_failed", message="Generator execution failed; inspect private worker metadata")
        status = Status.SUCCESS if error is None else Status(error.code) if error.code in Status._value2member_map_ else Status.GENERATION_FAILED
        return ModuleResult(module="generator", status=status, mock=request.mode == "mock", started_at=stamp,
            elapsed_ms=(perf_counter()-started)*1000, inputs=[a.artifact_id for a in input_artifacts],
            outputs=[candidate.artifact.artifact_id] if candidate else [], parameters=plan.parameters,
            data=metadata, error=error, candidate=candidate)
