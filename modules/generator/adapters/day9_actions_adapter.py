"""API-only validation; model Workers import the standard-library control loader."""
from typing import Literal
from pydantic import Field
from shared.schemas import Contract
from modules.generator.adapters.isolated import IsolatedGenerator
from modules.generator.adapters.day9_actions import VERSION, resolve


class Day9Parameters(Contract):
    prompt_id: str = Field(min_length=1, max_length=100)
    bank_version: Literal['liveportrait_action_tuning_v1_1028'] = VERSION


class Day9Generator(IsolatedGenerator):
    def _validated_parameters(self, request, plan):
        params = Day9Parameters.model_validate(plan.parameters).model_dump()
        resolve(params['prompt_id'], request.prompt)
        return params
