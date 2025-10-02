# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Code adapted from https://github.com/snu-mllab/KVzip

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def get_model_id(name: str):
    """ We support abbreviated model names such as:
        llama3.1-8b, llama3.2-*b, qwen2.5-*b, qwen3-*b, and gemma3-*b.
        The full model ID, such as "meta-llama/Llama-3.1-8B-Instruct", is also supported.
    """

    size = name.split("-")[-1].split("b")[0]  

    if name == "llama3.1-8b":
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif name == "llama3.0-8b":
        return "meta-llama/Meta-Llama-3-8B-Instruct"
    elif name.startswith("llama3.2-"):
        assert size in ["1", "3"], "Model is not supported!"
        return f"meta-llama/Llama-3.2-{size}B-Instruct"

    elif name.startswith("qwen2.5-"):
        assert size in ["7", "14"], "Model is not supported!"
        return f"Qwen/Qwen2.5-{size}B-Instruct-1M"
    else:
        return name  # Warning: some models might not be compatible and cause errors


def get_dtype(name: str):
    if name == "fp32":
        return torch.float32
    elif name == "fp16":
        return torch.float16
    elif name == "bf16":
        return torch.bfloat16
    else:
        return None


def load_model(model_name: str, dtype=None, **kwargs):
    model_id = get_model_id(model_name)
    from model.monkeypatch import replace_attn
    replace_attn(model_id)

    config = AutoConfig.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=config.torch_dtype if dtype is None else get_dtype(dtype),
        device_map="auto",
        attn_implementation='flash_attention_2',
        config=config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if "llama" in model_id.lower():
        model.generation_config.pad_token_id = tokenizer.pad_token_id = 128004

    if "gemma-3" in model_id.lower():
        model = model.language_model
    

    model.eval()
    model.name = model_name.split("/")[-1]
    print(f"\nLoad {model_id} with {model.dtype}")
    return model, tokenizer

