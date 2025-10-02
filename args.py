# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument(
    '--level',
    type=str,
    default='pair',
    choices=['pair', 'head', 'pair-uniform'],
    help="head: context-independent head-level eviction. pair-uniform: uniform head-budget ratios")
parser.add_argument(
    '-m',
    '--model',
    type=str,
    help=
    "check the model list in model/load.py"
)
parser.add_argument('--dtype', type=str, default=None, help="model dtype (automatically loaded)")
parser.add_argument('-d',
                    '--data',
                    type=str,
                    help="check the dataset list in data/load.py (e.g., squad, needle, scbench_kv, locomo)")

# Block prefill eviction
parser.add_argument('--prefill_chunk_size', type=int, default=2048, help="the chunk size for prefill")
parser.add_argument('--exp_name', type=str, default="", help="experiment name")

parser.add_argument('--verbose', action="store_true")
parser.add_argument('--kv_budget', type=int, default=8192, help="KV budget")
parser.add_argument(
    '--scoring_method',
    type=str,
    default='clustering',
    choices=['clustering', 'snapkv', 'kvzip', 'infinipot', 'keydiff'],
)

# Option for clustering
parser.add_argument('--n_cluster', type=int, default=4, help="number of clusters")
parser.add_argument('--n_medoid', type=int, default=4, help="number of medoids")
parser.add_argument('--embedding_type', type=str, default="sentence", choices=["sentence", "llm", "qwen", "qwen-4B", "qwen-0.6B"], help="embedding type")
parser.add_argument('--conv_window', type=int, default=4, help="number of utterances per conversation window")
parser.add_argument('--score_path', type=str, default=None, help="score path")
parser.add_argument('--power', type=float, default=0, help="power")
parser.add_argument('--target_length', type=int, default=20000, help="target length")

args = parser.parse_args()
