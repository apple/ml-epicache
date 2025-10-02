# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import os
import torch
from datasets import load_dataset, Dataset
import json

def load_dataset_all(name, tokenizer, n_data=100, subtask=None, target_length=20000):
    """ 
    Each data example has a format of {context: str, question: List[str], answers: List[str]}
    
    possible datasets = ["locomo", "realtalk", "longmemeval"]
    """

    metric = None
    if name == "locomo":
        dataset = load_locomo()
    elif name == "realtalk":
        dataset = load_realtalk()
    elif name == "longmemeval":
        dataset = load_longmemeval(target_length=target_length)
    else:
        raise ValueError(f"Invalid dataset: {name}")

    print(f"\n{name} loaded, #data: {len(dataset)}")
    return dataset, metric

def load_locomo(path="./data/locomo"):
    """
    Load LocoMo dataset from preprocessed .pt file
    """
    dataset = []
    samples = torch.load(os.path.join(path, "locomo_preprocessed.pt"))
    for data in samples:
        d = {}
        d["context"] = data["prompts"][0]
        d["question"] = data["prompts"][1:]  # questions after context
        d["answers"] = []
        for gt in data["ground_truth"]:
            if isinstance(gt, list):
                gt = ", ".join(gt)
            else:
                gt = str(gt)
            d["answers"].append(gt)

        # Add task_types information        
        if "task_types" in data:
            d["task_types"] = data["task_types"]
        else:
            # d["task_types"] = [0] * len(d["answers"])  # default to 0 if not available
            raise ValueError("task_types not available")

        dataset.append(d)

    return dataset

def load_realtalk(path="./data/realtalk"):
    """
    Load RealTalk dataset from preprocessed .pt file
    """
    dataset = []
    samples = torch.load(os.path.join(path, "realtalk_preprocessed.pt"))

    for data in samples:
        d = {}
        d["context"] = data["prompts"][0]
        d["question"] = data["prompts"][1:]  # questions after context
        d["answers"] = []
        for gt in data["ground_truth"]:
            if isinstance(gt, list):
                gt = ", ".join(gt)
            else:
                gt = str(gt)
            d["answers"].append(gt)

        # Add task_types information        
        if "task_types" in data:
            d["task_types"] = data["task_types"]
        else:
            raise ValueError("task_types not available")

        dataset.append(d)

    return dataset

def load_longmemeval(path="./data/longmemeval", target_length=20000):
    """
    Load LongMemEval dataset from preprocessed .pt file
    """
    dataset = []
    # samples = torch.load(os.path.join(path, "longmemeval_100conv_scbench.pt"))
    samples = torch.load(os.path.join(path, f"custom_lme_{target_length}_50.pt"))

    for data in samples:
        d = {}
        d["context"] = data["prompts"][0]
        d["question"] = data["prompts"][1:]  # questions after context
        d["answers"] = []
        for gt in data["ground_truth"]:
            if isinstance(gt, list):
                gt = ", ".join(gt)
            else:
                gt = str(gt)
            d["answers"].append(gt)

        # Add task_types information        
        if "task_types" in data:
            d["task_types"] = data["task_types"]
        else:
            raise ValueError("task_types not available")

        # Add metadata for LongMemEval
        if "conversation_id" in data:
            d["conversation_id"] = data["conversation_id"]
        if "total_tokens" in data:
            d["total_tokens"] = data["total_tokens"]

        dataset.append(d)

    return dataset