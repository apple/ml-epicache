# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Code adapted from https://github.com/snu-mllab/KVzip

import torch
from transformers import DynamicCache, HybridCache
from attention.score import KVScore
from typing import Tuple, Optional, Dict, Any

from tiny_api_cuda import update_flatten_view
    
class EvictCache(DynamicCache, KVScore):
    """ KV cache that evicts KV from the cache before decoding.
    """

    def __init__(self, model, evict_range: Tuple[int, int], kv_budget=16000):
        DynamicCache.__init__(self)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.n_heads_kv = model.config.num_key_value_heads
        self.n_group_kv = self.n_heads // self.n_heads_kv
        self.kv_budget = kv_budget

        self.start_idx, self.end_idx = evict_range
        self.ctx_len = self.end_idx - self.start_idx
        self.sink = self.start_idx  # retain initial KV pairs for system prompts
        self.prefill_ids = None
        self.ctx_ids = None

        self.get_score = False  # indicator for KV scoring
        self.evicted = False  # whether KV cache is evicted or not

        self.valid_pad = torch.ones((1, self.n_heads_kv, self.start_idx),
                                    dtype=bool,
                                    device=self.device)
        self.budget = None
        self.info = {"flatten": False, "offset": None}

    def update(self,
               key_states: torch.Tensor,
               value_states: torch.Tensor,
               layer_idx: int,
               cache_kwargs=dict()):
        """ Update KV cache and return 
        """
        if layer_idx == 0:
            seen_token = cache_kwargs.get("seen_token", key_states.size(-2))
            self._seen_tokens += seen_token

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        elif self.info["flatten"]:
            
            cu_klen = self.info["cu_len_k"][layer_idx]
            if self.prefill_mode:
                head_lens = self.info["len_k"][layer_idx] + key_states.shape[2]
            else:
                head_lens = self.info["len_k"][layer_idx] + self.info["offset"][layer_idx]

            dim = key_states.size(-1)

            if key_states.shape[2] == 1: # use adapted kernel for auto-regressive generation

                self.key_cache[layer_idx] = update_flatten_view(
                    self.key_cache[layer_idx],
                    key_states.contiguous().view(-1, dim),
                    head_lens,
                    cu_klen,
                )
                self.value_cache[layer_idx] = update_flatten_view(
                    self.value_cache[layer_idx],
                    value_states.contiguous().view(-1, dim),
                    head_lens,
                    cu_klen,
                )

            else:
                # concat flattened kv embedding with meta data using Python
                bsz, n_heads_kv, seq_len, dim = key_states.shape
                
                # Get current head lengths (excluding the new tokens)
                current_head_lens = self.info["len_k"][layer_idx]
                current_cu_lens = self.info["cu_len_k"][layer_idx]
                
                # Calculate new head lengths after concatenation
                new_head_lens = current_head_lens + seq_len
                new_total_tokens = new_head_lens.sum().item()
                
                # Create new flattened cache tensors
                new_key_cache = torch.empty(new_total_tokens, dim, 
                                          device=key_states.device, dtype=key_states.dtype)
                new_value_cache = torch.empty(new_total_tokens, dim, 
                                            device=value_states.device, dtype=value_states.dtype)
                
                # Concat each head separately
                new_offset = 0
                for head_idx in range(n_heads_kv):
                    # Current head length and position in old cache
                    curr_head_len = current_head_lens[head_idx].item()
                    curr_start_pos = current_cu_lens[head_idx].item()
                    
                    # Copy existing cache for this head
                    if curr_head_len > 0:
                        new_key_cache[new_offset:new_offset + curr_head_len] = \
                            self.key_cache[layer_idx][curr_start_pos:curr_start_pos + curr_head_len]
                        new_value_cache[new_offset:new_offset + curr_head_len] = \
                            self.value_cache[layer_idx][curr_start_pos:curr_start_pos + curr_head_len]
                
                    # Append new tokens for this head
                    new_key_cache[new_offset + curr_head_len:new_offset + curr_head_len + seq_len] = \
                        key_states[0, head_idx, :, :]  # assuming bsz=1
                    new_value_cache[new_offset + curr_head_len:new_offset + curr_head_len + seq_len] = \
                        value_states[0, head_idx, :, :]  # assuming bsz=1
                    
                    new_offset += new_head_lens[head_idx].item()
                
                # Update cache
                self.key_cache[layer_idx] = new_key_cache
                self.value_cache[layer_idx] = new_value_cache
                    
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def slice_flattened(self, query_length: int, seen_token_prev: int):
        """ Remove query_length tokens from the end of each head's flattened cache
        """
        for layer_idx in range(self.n_layers):

            cu_klen = self.info["cu_len_k"][layer_idx]
            head_lens = self.info["len_k"][layer_idx]
            
            # Keep only (head_lens - query_length) tokens for each head
            new_head_lens = torch.clamp(head_lens - query_length, min=0)

            if self.info["flatten"]:
                # Keep only (head_lens - query_length) tokens for each head
                self.key_cache[layer_idx] = torch.cat([
                    self.key_cache[layer_idx][cu_klen[h]:cu_klen[h] + new_head_lens[h]]
                    for h in range(self.n_heads_kv) if new_head_lens[h] > 0
                ])
                
                self.value_cache[layer_idx] = torch.cat([
                    self.value_cache[layer_idx][cu_klen[h]:cu_klen[h] + new_head_lens[h]]
                    for h in range(self.n_heads_kv) if new_head_lens[h] > 0
                ])
                
                if hasattr(self.info, "offset"):
                    self.info["offset"][layer_idx] -= query_length
            else:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :-query_length, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :-query_length, :]
            
            # Update metadata
            self.info["len_k"][layer_idx] = new_head_lens
            # self.info["concat_len_k"][layer_idx] = new_head_lens
            self.info["cu_len_k"][layer_idx] = torch.cat([
                torch.tensor([0], dtype=torch.int32, device=self.device),
                new_head_lens.cumsum(0)
            ])
        self._seen_tokens = seen_token_prev

    def slice(self, seen_token_prev: int):
        """ Evict KV of qeuries and generated tokens from the cache (for the reuse of the context cache)
        """
        for layer_idx in range(self.n_layers):
            if self.info["flatten"]:
                cu_klen = self.info["cu_len_k"][layer_idx]
                head_lens = self.info["len_k"][layer_idx]

                self.key_cache[layer_idx] = torch.cat([
                    self.key_cache[layer_idx][cu_klen[h]:cu_klen[h] + head_lens[h]]
                    for h in range(self.n_heads_kv)
                ])
                self.value_cache[layer_idx] = torch.cat([
                    self.value_cache[layer_idx][cu_klen[h]:cu_klen[h] + head_lens[h]]
                    for h in range(self.n_heads_kv)
                ])

                self.info["cu_len_k"][
                    layer_idx] -= self.info["offset"][layer_idx] * self.info["cu_head"]
            else:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :seen_token_prev]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :seen_token_prev]

        self.info["offset"] = [0 for _ in range(self.n_layers)]
        self._seen_tokens = seen_token_prev

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        elif self.info["flatten"]:
            # return self.info["full_len"] + self.info["offset"][layer_idx]
            return self._seen_tokens
        else:
            return self.key_cache[layer_idx].size(-2)

    def _mem(self):
        """ Returns the memory usage of the cache in GB.
        """
        if self.info["flatten"]:
            mem = 0
            for i in range(self.n_layers):
                mem += self.key_cache[i].numel() * self.key_cache[i].element_size()
        else:
            mem = self.n_layers * self.key_cache[0].numel() * self.key_cache[0].element_size()

        mem *= 2  # key + value
        return round(mem / 10**9, 4)

    def _budget(self):
        """ Returns the budget for each layer
        """
        key_budget_list = torch.zeros(self.n_layers, device=self.device)
        for layer_idx in range(self.n_layers):
            key_budget = self.key_cache[layer_idx].shape[0] // self.n_heads_kv
            key_budget_list[layer_idx] = key_budget
        return key_budget_list
    
    def evict_flattened(self, level: str = "pair"):
        """ Prune the flattened KV cache using flattened scores
        """
        if not hasattr(self, 'flattened_scores') or self.flattened_scores is None:
            raise ValueError("No flattened scores available for pruning")
        
        # Concatenate all scores across layers for processing
        all_layer_scores = [torch.cat(self.flattened_scores[i], dim=0) if self.flattened_scores[i] 
                           else torch.empty(0, device=self.device, dtype=self.dtype) 
                           for i in range(self.n_layers)]

        if "uniform" in level:
            # Uniform pruning: maintain equal ratios per head using 2D reshape
            layer_valids = []
            for layer_idx in range(self.n_layers):
                
                layer_scores = all_layer_scores[layer_idx]
                
                # Reshape to [n_heads_kv, head_len] for vectorized processing
                len_k = self.info["len_k"][layer_idx]
                head_len = len_k[0].item()  # All heads have same length in uniform case
                
                # Reshape scores to 2D: [n_heads_kv, head_len]
                scores_2d = layer_scores.view(self.n_heads_kv, head_len)
                
                if self.budget is not None: # Budtet Allocation
                    kv_budget = self.budget[layer_idx]
                else:
                    kv_budget = self.kv_budget

                if head_len > kv_budget:
                    k = kv_budget
                    _, topk_indices = torch.topk(scores_2d, min(k, head_len), dim=-1)
                    
                    # Create valid mask using scatter
                    valid_2d = torch.zeros_like(scores_2d, dtype=bool)
                    valid_2d.scatter_(-1, topk_indices, True)
                else:
                    valid_2d = torch.ones_like(scores_2d, dtype=bool)
                
                # Flatten back to 1D
                layer_valid = valid_2d.view(-1)
                layer_valids.append(layer_valid)
        
        else:
            # Head-wise (non-uniform) pruning: global top-k selection regardless of head
            layer_valids = []
            for layer_idx in range(self.n_layers):
                layer_scores = all_layer_scores[layer_idx]

                if self.budget is not None: # Budget Allocation
                    kv_budget_flattened = self.budget[layer_idx] * self.n_heads_kv
                else:
                    kv_budget_flattened = self.kv_budget * self.n_heads_kv
                    
                if len(layer_scores) > kv_budget_flattened:
                    top_k_indices = torch.topk(layer_scores, kv_budget_flattened).indices
                    layer_valid = torch.zeros(layer_scores.shape, dtype=torch.bool, device=layer_scores.device)
                    layer_valid[top_k_indices] = True
                    assert layer_valid.sum() == kv_budget_flattened, "Invalid mask"
                else:
                    layer_valid = torch.ones_like(layer_scores, dtype=bool)
                
                layer_valids.append(layer_valid)
        
        self.valid = layer_valids
        
        # Calculate statistics
        total_tokens = sum(len(scores) for scores in all_layer_scores)
        kept_tokens = sum(valid.sum().item() for valid in layer_valids)
        actual_ratio = kept_tokens / total_tokens
        removed_tokens = total_tokens - kept_tokens
        
        self.prepare_block()
        self.evicted = True

    def _get_valid(self, layer_idx: int, n_seq: int):
        """ obtain full mask for the given keys (retain system prompt and queries)
        """
        valid = torch.cat([self.valid_pad, self.valid[layer_idx]], dim=-1)  # sys prompt + context

        size = list(valid.shape)
        size[-1] = n_seq - valid.shape[-1]
        ones = torch.ones(size, device=valid.device, dtype=bool)
        valid = torch.cat([valid, ones], dim=-1)  # sys prompt + context + query ...

        return valid

    def prepare_block(self):
        """ Evict KV and prepare metadata for FlashAttention with head-wise compression
        """
        len_k_layers = []
        max_len_k_layers = []
        cu_len_k_layers = []

        for layer_idx in range(self.n_layers):
            # Get valid mask with system token preservation
            valid = self.valid[layer_idx]

            # Flatten and apply valid mask
            if len(self.key_cache[layer_idx].shape) == 4:
                _, _, klen, dim = self.key_cache[layer_idx].shape
                self.key_cache[layer_idx] = self.key_cache[layer_idx].contiguous().view(-1, dim)[valid.view(-1)]
                self.value_cache[layer_idx] = self.value_cache[layer_idx].contiguous().view(-1, dim)[valid.view(-1)]
            else:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][valid]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][valid]
                
            original_len_k = self.info["len_k"][layer_idx]
            lens_k_head = []
            start_idx = 0
            
            # Head-wise key length
            for head_idx in range(self.n_heads_kv):
                head_len = original_len_k[head_idx].item()
                head_valid = valid[start_idx:start_idx + head_len]
                lens_k_head.append(head_valid.sum().item())
                start_idx += head_len
        
            lens_k_head = torch.tensor(lens_k_head, dtype=torch.int32, device=self.device)
    
            # Calculate cumulative lengths
            cu_seqlens_k = lens_k_head.cumsum(0).int()
            cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32, device=self.device), cu_seqlens_k])

            len_k_layers.append(lens_k_head)
            max_len_k_layers.append(lens_k_head.max())
            cu_len_k_layers.append(cu_seqlens_k)
        
        cu_head = torch.arange(self.n_heads_kv + 1, dtype=torch.int32, device=self.device)
        
        self.info = {
            "flatten": True,
            "cu_head": cu_head,
            "len_k": len_k_layers,  # kv lengths of heads in a layer
            "max_len_k": max_len_k_layers,  # max kv lengths of heads in a layer
            "cu_len_k": cu_len_k_layers,  # cumulative kv length of heads in a layer (only updated)
            "full_len": 0,  # not used in compressed case
            "offset": [0 for _ in range(self.n_layers)],  # newly processed kv lengths
        }
    
    def prepare(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        prefill_mode: bool = False,
    ):
        """ Subsample KV and flatten features for var_len FlashAttention
        """
        bsz, n_heads_q, q_len, dim = query_states.shape

        query_states = query_states.view(bsz, self.n_heads_kv, self.n_group_kv, q_len, dim)
        query_states = query_states.transpose(2, 3).contiguous().view(
            -1, self.n_group_kv, dim)  # bsz x head x seq, group, dim

        self.info["offset"][layer_idx] += q_len
        self.info["cu_len_k"][layer_idx] += q_len * self.info["cu_head"]

        if prefill_mode:
            self.info["len_k"][layer_idx] = self.info["len_k"][layer_idx] + q_len
        
        info = {
            "cu_len_q": q_len * self.info["cu_head"],
            "cu_len_k": self.info["cu_len_k"][layer_idx],
            "max_len_q": q_len,
            "max_len_k": self.info["max_len_k"][layer_idx] + self.info["offset"][layer_idx]
        }

        return query_states, key_states.view(-1, 1, dim), value_states.view(-1, 1, dim), info

    def to_cpu(self):
        """ Move key/value caches and info tensors to CPU
        """
        # Move key and value caches to CPU
        for layer_idx in range(self.n_layers):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].cpu()
            self.value_cache[layer_idx] = self.value_cache[layer_idx].cpu()
        
        # Move all tensor values in info dict to CPU
        for key, value in self.info.items():
            if isinstance(value, list):
                self.info[key] = [v.cpu() if torch.is_tensor(v) else v for v in value]
            elif torch.is_tensor(value):
                self.info[key] = value.cpu()
        
        self.device = torch.device('cpu')
        return self

    def to_gpu(self, device=None):
        """ Move key/value caches and info tensors to GPU
        """
        # Move key and value caches to GPU
        for layer_idx in range(self.n_layers):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device)
        
        # Move all tensor values in info dict to GPU
        for key, value in self.info.items():
            if isinstance(value, list):
                self.info[key] = [v.to(device) if torch.is_tensor(v) else v for v in value]
            elif torch.is_tensor(value):
                self.info[key] = value.to(device)
        
        self.device = device
        return self

