# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Code adapted from https://github.com/snu-mllab/KVzip

def template(model_name, task):
    model_name = model_name.lower()

    if "llama" in model_name:
        prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prefix += "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"

        postfix = "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif "mistral" in model_name:
        prefix = "<s>[INST]"
        postfix = "[/INST]"

    elif "qwen2.5-" in model_name:
        # https://github.com/QwenLM/Qwen3
        prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prefix += "<|im_start|>user\n"

        postfix = "<|im_end|>\n<|im_start|>assistant\n"
    else:
        raise ValueError(f"Model {model_name} is not supported")

    return prefix, postfix
