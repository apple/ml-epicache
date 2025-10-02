# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Code adapted from https://github.com/snu-mllab/KVzip

import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union, Optional


class KVScore():
    """ Functions to compute the score for the KV features. (kvcache.py)"""

    def __init__(self):
        self.n_heads_kv = None
        self.dtype = None
        self.device = None
        self.get_score = True
        self.causal_mask_score = None
        self.score = None
        self.sink = None
        self.start_idx, self.end_idx = None, None

    def init_score(self):
        self.get_score = True
        self.causal_mask_score = None
        self.score = [
            torch.zeros((1, self.n_heads_kv, 0), dtype=self.dtype, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.flattened_scores = [[] for _ in range(self.n_layers)]

    def _update_score(self, layer_idx: int, score: torch.Tensor):
        self.score[layer_idx] = torch.cat([self.score[layer_idx], score], dim=-1)

    def _get_score(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """ Compute KV importance scores.
            # key_states: bsz x head_kv x k x dim, query_states: bsz x head x q x dim
        """

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv = key_states.size(1)

        query_states = query_states.view(bsz, num_kv, -1, q_len, head_dim)
        key_states = torch.cat(
            [
                key_states[:, :, :self.sink],  # sink tokens (generally system prompt)
                key_states[:, :, self.start_idx:self.end_idx],  # KV chunk in the cache
                key_states[:, :, -q_len:],  # KV repeat chunk
            ],
            dim=2)

        # bsz, head, 1, dim, k
        key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()
        ctx_len = self.end_idx - self.start_idx

        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)
        self._mask_causal(attn_weights, q_len)

        # bsz, head, group, q, ctx_len
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)  # not fp32
        attn_weights = attn_weights[..., self.sink:self.sink + ctx_len]
        score = attn_weights.amax(dim=(-3, -2))  # max over group, q

        self._update_score(layer_idx, score)
    
    def _get_score_flatten(self, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int):
        """ Compute KV importance scores with flattened KV. (head-wise variable length scoring)
            We do not concat the score, but compute head-wise score per each function call.
            # key_states: bsz x head_kv x k x dim, query_states: bsz x head x q x dim
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        n_heads_kv = self.n_heads_kv

        # Reshape query_states to handle grouped query attention
        query_states = query_states.view(bsz, n_heads_kv, -1, q_len, head_dim)

        # Get flattened keys from cache
        if len(self.key_cache[layer_idx].shape) != 2:
            # First block prefill, key_states is not flattened
            flattened_keys = self.key_cache[layer_idx].contiguous().view(-1, key_states.size(-1))
            # Create temporary metadata for 4D tensor (all heads have same length)
            _, n_heads_kv, seq_len, _ = self.key_cache[layer_idx].shape
            cu_len_k = torch.arange(n_heads_kv + 1, dtype=torch.int32, device=self.device) * seq_len
            len_k = torch.full((n_heads_kv,), seq_len, dtype=torch.int32, device=self.device)
            
            # Initialize metadata lists for all layers with same values (score target length)
            if not hasattr(self, 'info') or self.info is None:
                self.info = {}

            self.info["cu_len_k"] = [cu_len_k for _ in range(self.n_layers)]
            self.info["len_k"] = [len_k for _ in range(self.n_layers)]

        else:
            flattened_keys = self.key_cache[layer_idx]
            # Get flattened KV cache metadata for this layer
            cu_len_k = self.info["cu_len_k"][layer_idx]  # cumulative lengths [n_heads_kv + 1]
            len_k = self.info["len_k"][layer_idx]        # actual lengths per head [n_heads_kv]
            len_k = len_k + q_len
            cu_len_k = torch.cat([torch.tensor([0], dtype=torch.int32, device=self.device),len_k.cumsum(0)])
            
        # Collect scores for each head
        head_scores = []
        
        for head_idx in range(n_heads_kv):
            
            # Get past key length (excluding current query's keys)
            total_kv_len = len_k[head_idx].item()
            past_kv_len = total_kv_len - q_len

            if past_kv_len <= 0:
                # Skip if no past keys available
                raise ValueError(f"No past keys available for head {head_idx}")
                
            # Extract past keys for this head: [past_kv_len, head_dim]            
            start_idx = cu_len_k[head_idx].item()
            head_keys = flattened_keys[start_idx:start_idx + past_kv_len, :]
            
            # Extract queries for this head: [bsz, num_group, q_len, head_dim]
            head_queries = query_states[:, head_idx, :, :, :]
            
            # Compute attention weights: [bsz, num_group, q_len, past_kv_len]
            attn_weights = torch.matmul(head_queries, head_keys.transpose(-2, -1)) / math.sqrt(head_dim)

            # Apply softmax: [bsz, num_group, q_len, past_kv_len]
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            
            # Compute importance score: [past_kv_len]
            head_score = attn_weights.amax(dim=(0, 1, 2))
            
            head_scores.append(head_score)

        # Concatenate and store scores
        if head_scores:
            all_scores = torch.cat(head_scores, dim=0) 
            self.flattened_scores[layer_idx].append(all_scores)

    def _make_mask(self, attn_weights: torch.Tensor, window_size: int):
        """ Define causal mask shared across layers
        """
        mask = torch.full((window_size, window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        self.causal_mask_score = mask[None, None, None, :, :]

    def _mask_causal(self, attn_weights: torch.Tensor, window_size: int):
        """ Apply causal maksing
        """
        if self.causal_mask_score is None:
            self._make_mask(attn_weights, window_size)
        elif self.causal_mask_score.size(-1) != window_size:
            self._make_mask(attn_weights, window_size)

        attn_weights[..., -window_size:, -window_size:] += self.causal_mask_score
