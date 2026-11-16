"""Author-recommended Diffusers 0.25.1 path, using local FP16 weights.

https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
"""
def generate(job, failure):
    import hashlib
    import json
    from pathlib import Path
    import diffusers
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from PIL import Image, ImageOps
    import torch
    from transformers import CLIPTokenizer
    from long_prompt import encode_clip_chunks, pack_clip_tokens
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    parameters = job["parameters"]
    encoding=parameters.get('prompt_encoding','native')
    tokenizer=CLIPTokenizer.from_pretrained(str(Path(job['checkpoint'])/'tokenizer'),local_files_only=True)
    tokens=tokenizer(job['prompt'],truncation=False)['input_ids']
    if encoding=='native':
        if len(tokens)>tokenizer.model_max_length:
            raise failure('invalid_input')
    elif encoding=='chunked_clip_v1':
        try:
            pack_clip_tokens(tokenizer(job['prompt'],add_special_tokens=False,truncation=False)['input_ids'],
                bos=tokenizer.bos_token_id,eos=tokenizer.eos_token_id,pad=tokenizer.pad_token_id)
        except ValueError:
            raise failure('invalid_input')
    else:
        raise failure('invalid_input')
    with Image.open(job["inputs"]["image"]) as original:
        original_size = original.size
        image = ImageOps.fit(ImageOps.exif_transpose(original).convert("RGB"), (512,512), method=Image.Resampling.LANCZOS)
    with torch.inference_mode():
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(job["checkpoint"],
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True,
            tokenizer=tokenizer).to("cuda")
        if pipe.safety_checker is None:
            raise failure("generation_failed")
        if encoding=='chunked_clip_v1':
            positive,negative,encoding_metadata=encode_clip_chunks(pipe,job['prompt'])
            text_inputs={'prompt_embeds':positive,'negative_prompt_embeds':negative}
        else:
            text_inputs={'prompt':job['prompt']}
            encoding_metadata={'mode':'native','input_tokens':len(tokens),'truncated':False,
                'prompt_sha256':hashlib.sha256(job['prompt'].encode()).hexdigest()}
        receipt=Path(job['output']).with_name('prompt_encoding.json')
        with receipt.open('x') as stream:stream.write(json.dumps(encoding_metadata,indent=2)+'\n')
        receipt.chmod(0o600)
        result = pipe(**text_inputs, image=image,
            generator=torch.Generator(device="cpu").manual_seed(parameters["seed"]),
            num_inference_steps=parameters["num_inference_steps"], guidance_scale=parameters["guidance_scale"],
            image_guidance_scale=parameters["image_guidance_scale"])
        if result.nsfw_content_detected is None or result.nsfw_content_detected[0]:
            raise failure("safety_filtered")
        result.images[0].save(job["output"])
    torch.cuda.synchronize()
    return {"torch": torch.__version__, "diffusers": diffusers.__version__, "input_size": list(original_size),
        "prompt_encoding":encoding_metadata,
        "preprocess": "EXIF transpose, RGB, centered aspect-preserving fit/crop to 512x512 (LANCZOS)",
        "scheduler": type(pipe.scheduler).__name__, "safety_checker_preserved": True,
        "nsfw_content_detected": False, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
