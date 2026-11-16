"""Official FFHQ StyleGAN2 through the ADA PyTorch implementation.

Source: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/generate.py
Only the server's trusted local checkpoint may be unpickled.
"""
import sys


def generate(job, failure):
    import numpy as np
    from PIL import Image
    import torch
    sys.path.insert(0, job["repo"])
    import legacy
    from torch_utils.ops import bias_act, upfirdn2d
    # Official reference operators; avoid compiling extensions in preserved environments.
    for module, name in ((bias_act, "bias_act"), (upfirdn2d, "upfirdn2d")):
        original = getattr(module, name)
        def reference(*args, _function=original, **kwargs):
            kwargs["impl"] = "ref"
            return _function(*args, **kwargs)
        setattr(module, name, reference)
    torch.set_num_threads(4)
    parameters = job["parameters"]
    torch.manual_seed(parameters["seed"])
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        with open(job["checkpoint"], "rb") as handle:
            generator = legacy.load_network_pkl(handle)["G_ema"].eval().requires_grad_(False).to("cuda")
        if generator.c_dim != 0:
            raise failure("generation_failed")
        latent = torch.from_numpy(np.random.RandomState(parameters["seed"]).randn(1,generator.z_dim)).to("cuda")
        image = generator(latent, torch.zeros((1,0), device="cuda"),
            truncation_psi=parameters["truncation_psi"], noise_mode=parameters["noise_mode"])
        if not torch.isfinite(image).all():
            raise failure("generation_failed")
        pixels = (image.permute(0,2,3,1)*127.5+128).clamp(0,255).to(torch.uint8)[0].cpu().numpy()
        Image.fromarray(pixels, "RGB").save(job["output"])
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "numpy": np.__version__, "z_dim": generator.z_dim,
        "c_dim": generator.c_dim, "training": False, "operators": "official PyTorch reference on CUDA",
        "checkpoint_scope": "original StyleGAN2 FFHQ, not ADA training",
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
