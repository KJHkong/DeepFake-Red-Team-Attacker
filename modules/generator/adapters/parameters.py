"""Bounded model parameters, validated before starting a worker process."""
from typing import Literal

from pydantic import ConfigDict, Field

from shared.schemas import Contract


class InSwapperParameters(Contract):
    """Largest face, CPU provider, det512 and paste-back are fixed in v0.1."""


class SimSwapParameters(Contract):
    """Fixed official224/mask/logo, det512 confidence0.6; no user command options."""


class Day6DiffSwapParameters(Contract):
    seed: int = Field(default=42,ge=0,le=2**32-1)
    steps: Literal[200] = 200
    tgt_scale: Literal[0.0] = 0.0


class Day6REFaceParameters(Contract):
    seed: int = Field(default=42,ge=0,le=2**32-1)
    steps: Literal[50] = 50
    guidance_scale: Literal[3.5] = 3.5


class BlendFaceParameters(Contract):
    """Fixed author forward, InsightFace112/FFHQ256 and native learned-mask paste."""


class MobileFaceSwapParameters(Contract):
    """Official224/identity112/det640 threshold0.6 and native paste are fixed."""


class UniFaceParameters(Contract):
    """Official256 FP16 with seed0; audited FFHQ/parser paste is fixed."""


class LivePortraitStaticParameters(Contract):
    action_id: Literal['mouth_open_to_closed','mouth_closed_to_open','left_to_frontal',
        'left_to_right','right_to_frontal','right_to_left','frontal_to_left','frontal_to_right',
        'eyes_open_to_closed','eyes_closed_to_open']


class SeedParameters(Contract):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32-1)


class InstructDiffusionParameters(SeedParameters):
    steps: Literal[50] = 50
    cfg_text: Literal[5.0] = 5.0
    cfg_image: Literal[1.25] = 1.25
    resolution: Literal[512] = 512
    prompt_encoding: Literal['chunked_clip_v1'] = 'chunked_clip_v1'


class LEDITSParameters(SeedParameters):
    action_id: Literal['mouth_open_to_closed','mouth_closed_to_open','left_to_frontal',
        'left_to_right','right_to_frontal','right_to_left','frontal_to_left','frontal_to_right',
        'eyes_open_to_closed','eyes_closed_to_open']
    num_inversion_steps: Literal[50] = 50
    skip: Literal[0.1] = 0.1
    edit_guidance_scale: Literal[10.0] = 10.0
    edit_threshold: Literal[0.75] = 0.75


class InstructPix2PixParameters(SeedParameters):
    num_inference_steps: int = Field(default=30, ge=1, le=50)
    guidance_scale: float = Field(default=7.5, ge=1, le=15)
    image_guidance_scale: float = Field(default=1.5, ge=1, le=3)
    prompt_encoding: Literal['native','chunked_clip_v1'] = 'native'


class StyleGAN2Parameters(SeedParameters):
    truncation_psi: float = Field(default=0.7, ge=0, le=1)
    noise_mode: Literal["const", "random", "none"] = "const"


PARAMETERS = {"inswapper": InSwapperParameters, "instruct_pix2pix": InstructPix2PixParameters,
              "stylegan2_ada": StyleGAN2Parameters,"simswap":SimSwapParameters,
              "liveportrait_static":LivePortraitStaticParameters,
              "diffswap":Day6DiffSwapParameters,"reface":Day6REFaceParameters,"blendface":BlendFaceParameters,
              "instructdiffusion":InstructDiffusionParameters,"leditspp":LEDITSParameters,
              "mobilefaceswap":MobileFaceSwapParameters,"uniface":UniFaceParameters}
