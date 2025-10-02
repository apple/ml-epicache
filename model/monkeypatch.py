# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Code adapted from https://github.com/snu-mllab/KVzip

import transformers
from attention.attn import llama_flash_attn2_forward

def replace_attn(model_id):
    model_id = model_id.lower()
    if "mistral" in model_id:
        transformers.models.mistral.modeling_mistral.MistralAttention.forward = llama_flash_attn2_forward

    if "llama" in model_id:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_flash_attn2_forward

    elif "qwen2.5" in model_id:
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = llama_flash_attn2_forward
    
    else:
        raise ValueError(f"Model {model_id} is not supported")
