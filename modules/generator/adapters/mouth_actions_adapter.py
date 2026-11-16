"""Opt-in profile on the existing isolated Generator; no default registration."""
from typing import Literal
from pydantic import Field
from shared.schemas import Contract
from modules.generator.adapters.isolated import IsolatedGenerator
from modules.generator.adapters.mouth_actions import VERSION,resolve


class MouthParameters(Contract):
    prompt_id: str=Field(min_length=1,max_length=100)
    bank_version: Literal['liveportrait_mouth_bidirectional_v1_1028']=VERSION


class MouthGenerator(IsolatedGenerator):
    def _validated_parameters(self,request,plan):
        params=MouthParameters.model_validate(plan.parameters).model_dump()
        resolve(params['prompt_id'],request.prompt)
        return params
