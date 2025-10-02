# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import torch
import os
import json
from time import time

def save_result(args, dataname, outputs, idx, do_fullkv):
    path = f"./results/{args.exp_name}/{dataname}/{idx}_{args.model}"
    # if do_fullkv:
    #     path += "_fullkv"
    os.makedirs(path, exist_ok=True)

    tag = f"-{args.level}"
    with open(f"{path}/output{tag}.json", 'w') as f:
        json.dump(outputs, f, indent=4)


def inplace_softmax(x, dim=-1):
    max_vals, _ = x.max(dim=dim, keepdim=True)
    x.sub_(max_vals)  # For numerical stability
    x.exp_()
    sum_exp = x.sum(dim=dim, keepdim=True)
    x.div_(sum_exp)
    return x


def gmem(text="", print=True):
    _, total_mem = torch.cuda.mem_get_info(0)
    total_mem = total_mem / 1024**3
    allc_mem = torch.cuda.memory_allocated(0) / 1024**3
    msg = f"## {allc_mem:.2f}/{total_mem:.2f} GB, {text}"
    if print:
        print(msg)
    return allc_mem, total_mem


class TimeStamp():

    def __init__(self, verbose=True, precision=1, unit="s"):
        self.verbose = verbose
        self.precision = precision
        self.unit = unit
        self.set()

    def set(self):
        if self.verbose:
            torch.cuda.synchronize()
            self.start = time()

    def elapsed(self, denominator=1.0):
        # example implementation
        val = time() - self.start
        if self.unit == "ms":
            val *= 1000
        return round(val / denominator, self.precision)

    def __call__(self, msg="", denominator=1.0):
        if self.verbose:
            torch.cuda.synchronize()
            allc_mem, total_mem = gmem(print=False)
            tt = self.elapsed(denominator)
            print(f"## Time: {tt}{self.unit}. Mem: {allc_mem:.2f}/{total_mem:.2f} GB. [{msg}]")
            print(flush=True)
            self.set()
        return tt
