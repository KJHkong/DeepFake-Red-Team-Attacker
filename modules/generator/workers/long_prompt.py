"""Explicit bounded CLIP chunk conditioning for the existing isolated image Worker.

Diffusers0.25.1 InstructPix2Pix accepts matching prompt_embeds/negative_prompt_embeds.
This adapter extension resets CLIP positions per77-token chunk and concatenates
hidden states. It preserves tokens, not a guarantee of semantic edit completion.
"""
import hashlib


def pack_clip_tokens(payload, *, bos, eos, pad):
    if len(payload)>300 or any(type(token) is not int or token<0 for token in payload):
        raise ValueError('CLIP chunk token limit or token type is invalid')
    chunks=[payload[i:i+75] for i in range(0,len(payload),75)] or [[]]
    ids=[];masks=[]
    for chunk in chunks:
        ids.append([bos]+chunk+[eos]+[pad]*(75-len(chunk)))
        masks.append([1]*(len(chunk)+2)+[0]*(75-len(chunk)))
    return {'input_ids':ids,'attention_mask':masks,'payload_lengths':[len(x) for x in chunks]}


def encode_clip_chunks(pipe,prompt):
    import torch
    tokenizer=pipe.tokenizer
    if tokenizer.model_max_length!=77:
        raise ValueError('Unexpected CLIP context contract')
    payload=tokenizer(prompt,add_special_tokens=False,truncation=False)['input_ids']
    special=dict(bos=tokenizer.bos_token_id,eos=tokenizer.eos_token_id,pad=tokenizer.pad_token_id)
    positive=pack_clip_tokens(payload,**special);empty=pack_clip_tokens([],**special)
    device=pipe._execution_device
    def encode(packed):
        ids=torch.tensor(packed['input_ids'],dtype=torch.long,device=device)
        mask=torch.tensor(packed['attention_mask'],dtype=torch.long,device=device)
        kwargs={'attention_mask':mask} if getattr(pipe.text_encoder.config,'use_attention_mask',False) else {}
        states=pipe.text_encoder(ids,**kwargs)[0]
        return states.reshape(1,-1,states.shape[-1])
    prompt_embeds=encode(positive)
    negative=encode(empty).repeat(1,len(positive['input_ids']),1)
    if prompt_embeds.shape!=negative.shape:
        raise ValueError('Positive/negative conditioning shapes disagree')
    return prompt_embeds,negative,{'mode':'chunked_clip_v1','payload_tokens':len(payload),
        'context_per_chunk':77,'chunks':len(positive['input_ids']),'conditioning_tokens':prompt_embeds.shape[1],
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'truncated':False,
        'method':'per-chunk CLIP positions; concatenated hidden states; repeated empty negative conditioning',
        'semantic_edit_quality':'not_evaluated'}
