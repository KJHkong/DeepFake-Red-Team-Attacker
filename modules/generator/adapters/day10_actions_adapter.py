"""Only the new semantic name, using existing native absolute-yaw0 math."""
from typing import Literal
from shared.schemas import Contract
from modules.generator.adapters.isolated import IsolatedGenerator
from modules.generator.adapters.day10_actions import resolve,VERSION


class KeepFrontalParameters(Contract):
    prompt_id: Literal['keep_frontal_canonical_day10']='keep_frontal_canonical_day10'
    bank_version: Literal['liveportrait_keep_frontal_v1_1029']=VERSION


class KeepFrontalGenerator(IsolatedGenerator):
    def _validated_parameters(self,request,plan):
        params=KeepFrontalParameters.model_validate(plan.parameters).model_dump()
        resolve(params['prompt_id'],request.prompt)
        return params
