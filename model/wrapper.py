# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

# Code adapted from https://github.com/snu-mllab/KVzip

import torch
import time
import glob
import math
from typing import List, Tuple, Union, Optional
from tqdm import tqdm
from transformers import DynamicCache, Gemma3ForCausalLM, Qwen3ForCausalLM, MistralForCausalLM

from attention.kvcache import EvictCache
from model.load import load_model
from model.template import template
import json

class LongConvQAModel():

    def __init__(self, model_name: str, dtype: Optional[str] = None, evict_level: str = "pair", scoring_method: str = "kvzip"):
        self.model, self.tokenizer = load_model(model_name, dtype=dtype)

        self.name = self.model.name
        self.dtype = self.model.dtype
        self.device = self.model.device
        self.config = self.model.config
        self.evict_level = evict_level
        self.scoring_method = scoring_method
        self.cluster_ids = None
        self.snap_window_size = 64

        self.gen_kwargs = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1,
            "top_k": None,
            "max_new_tokens": 50,
        }
        
        self.set_chat_template()

    def allocate_layer_budget(self, score_dict, kv_budget=8192, power=2.0, retrieval=False):
        """
        Power-based budget allocation: Lower scores get more budget
        
        Args:
            score_dict: Dictionary containing 'combined_scores'
            kv_budget: Average budget per layer
            power: Controls allocation extremity (higher = more extreme)
        
        Returns:
            torch.Tensor: Budget per layer [num_layers]
        """
        scores = score_dict['combined_scores']
        
        # Convert to tensor if needed
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(scores, dtype=torch.float32)
        else:
            scores = scores.float()
        
        # Move to same device
        scores = scores.to(self.device)
        
        # Normalize scores to [0, 1] range
        if scores.max() > scores.min():
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            # All scores are the same, allocate equally
            return torch.full((len(scores),), kv_budget, dtype=torch.int, device=self.device)
        
        # Inverse with power (low score = high weight)
        inverse_weights = (1.0 - normalized_scores) ** power
        
        # Allocate budget maintaining average = kv_budget
        total_budget = kv_budget * len(scores)
        budget = inverse_weights * total_budget / inverse_weights.sum()
        
        # Convert to int and ensure minimum budget
        budget = budget.int()
        min_budget = max(1, kv_budget // 4)  # Minimum 25% of average budget
        budget = torch.clamp(budget, min=min_budget)
        
        return budget

    def encode(self, text: str) -> torch.Tensor:
        """ Encode text into tokens
        """
        return self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").cuda()

    def decode(self, input_ids: torch.Tensor) -> str:
        """ Decode tokens into text
        """
        if len(input_ids.shape) == 2:
            input_ids = input_ids[0]
        return self.tokenizer.decode(input_ids)

    def set_chat_template(self, task: str = "qa"):
        prefix, postfix = template(self.name, task)
        self.sys_prompt_ids, self.postfix_ids = self.encode(prefix), self.encode(postfix)

    def apply_template(self, query: str) -> torch.Tensor:
        query = f"\n\n{query.strip()}"
        query_ids = torch.cat([self.encode(query), self.postfix_ids], dim=1)
        return query_ids

    def __call__(
        self,
        input_ids: torch.Tensor,
        kv: Union[EvictCache],
        update_cache: bool = False,
        return_logits: bool = False,
        flatten_mode: bool = False,
        *args,
        **kwargs,
    ):
        """ Compute Transformer forward pass
            In default, we do not update the KV cache with the newly given inputs.
            Set update_cache = True to enable the update.
        """
        seen_token_prev = kv._seen_tokens

        if return_logits:
            outputs = self.model(input_ids, past_key_values=kv, *args, **kwargs)
        else:
            _ = self.model.model(input_ids, past_key_values=kv, *args, **kwargs)
            outputs = None

        if not update_cache:
            if flatten_mode: 
                kv.slice_flattened(input_ids.shape[1], seen_token_prev)
            else:
                kv.slice(seen_token_prev)
        return outputs

    def _init_kv(self, kv=None, evict_range=(0, 0), kv_budget=16000):
        """ Initialize KV cache
        """
        if kv is None:
            kv = EvictCache(self.model, evict_range, kv_budget)
        return kv

    @torch.inference_mode()
    def prefill_memory_constrained(
        self,
        ctx_ids: Union[str, torch.Tensor],
        prefill_chunk_size: int = 16000,
        kv_budget: int = 8192,
        load_score=False,
        do_score=True,
        score_path=None,
        power=2.0,
        question=None,
        pyramid=False,
        retrieval=False,
    ) -> Union[EvictCache]:
        """ Memory-constrained chunked prefill with adaptive budget allocation
        """
        assert do_score, "do_score must be True"
        
        if type(ctx_ids) == str:
            ctx_ids = self.encode(ctx_ids)
        prefill_ids = torch.cat([self.sys_prompt_ids, ctx_ids], dim=1)
        evict_range = (self.sys_prompt_ids.shape[1], prefill_ids.shape[1])

        kv = self._init_kv(evict_range=evict_range, kv_budget=kv_budget)  
        kv.prefill_mode = True
        
        if power > 0:
            # Load layer scores from JSON file
            with open(score_path, 'r') as f:
                score_dict = json.load(f)
            kv.budget = self.allocate_layer_budget(score_dict, kv_budget, power=power, retrieval=retrieval)
        
        kv.ctx_ids = ctx_ids
        kv.prefill_ids = prefill_ids
        
        # prefill with adaptive budget allocation
        cur_pos = 0
        total_length = prefill_ids.shape[1]
        
        # Simple chunk processing - just split by prefill_chunk_size
        while cur_pos < total_length:
            # Calculate chunk size - simply use prefill_chunk_size
            remaining_tokens = total_length - cur_pos
            chunk_size = min(prefill_chunk_size, remaining_tokens)
            
            # Get current chunk
            input_ids = prefill_ids[:, cur_pos:cur_pos + chunk_size]
            
            # Calculate cache position for current chunk
            chunk_cache_position = torch.arange(
                cur_pos, 
                cur_pos + chunk_size, 
                device=input_ids.device
            )
            # 1. Process current chunk
            self.__call__(input_ids, kv, update_cache=True, cache_position=chunk_cache_position)
            # 2. KV importance scoring 
            self.token_scoring(kv, input_ids, load_score=load_score, cache_position=chunk_cache_position, question=question)
            # 3. KV eviction based on importance score
            kv.evict_flattened(self.evict_level)  
            cur_pos += chunk_size
        
        kv.prefill_mode = False
        return kv

    @torch.inference_mode()
    def token_scoring(
        self,
        kv: Union[EvictCache],
        ctx_ids: torch.Tensor,
        load_score=False,
        cache_position=None,        
        question=None,
    ):
        """ KV importance scoring with patched prompt
        """
        kv.init_score()

        if self.scoring_method in ["kvzip", "infinipot", "clustering", "snapkv"]:
            if self.scoring_method == "kvzip":
                kv.end_idx = ctx_ids.shape[1]
                prompt = f"\n\nRepeat the part of the previous context exactly."
                q_ids = self.encode(prompt)
                repeat_ids = torch.cat([q_ids, self.postfix_ids, ctx_ids], dim=1)
            
            elif self.scoring_method == "infinipot":
                prompt = f"\n\nSummarize the previous context highlighting the most important parts."
                q_ids = self.encode(prompt)
                repeat_ids = torch.cat([q_ids, self.postfix_ids], dim=1)
            
            elif self.scoring_method == "clustering":
                prompt = f"\n\nExtract essential information from this conversation segment. Identify speaker names, personas, key events with dates/times, personal details, commitments, opinions, relationships, emotional moments, and unique experiences."
                q_ids = self.encode(prompt)
                repeat_ids = torch.cat([q_ids, self.postfix_ids, self.cluster_ids], dim=1)
            
            elif self.scoring_method == "snapkv": # use the last window of ctx_ids as a patched prompt
                repeat_ids = ctx_ids[:, -self.snap_window_size:]
            else:
                raise NotImplementedError(f"Scoring method {self.scoring_method} is not implemented")

            repeat_cache_position = torch.arange(cache_position[-1], cache_position[-1] + repeat_ids.shape[1], device=repeat_ids.device)
            self.__call__(repeat_ids, kv, update_cache=False, cache_position=repeat_cache_position, flatten_mode=True)
            
            kv.get_score = False 
        
        elif self.scoring_method in ["keydiff"]:

            for layer_idx in range(kv.n_layers):
                key_states = kv.key_cache[layer_idx]
                key_states_norm = key_states / key_states.norm(dim=-1, keepdim=True)
                
                # Compute anchor vector
                if len(key_states.shape) == 4:  # (batch, heads, seq_len, dim)
                    # Average over sequence dimension to get anchor
                    anchor_key = key_states_norm.mean(dim=2, keepdim=True)  # (batch, heads, 1, dim)
                else:  # (seq_len, dim) or similar
                    # Average over sequence dimension
                    anchor_key = key_states_norm.mean(dim=0, keepdim=True)  # (1, dim)
                
                # Compute cosine similarity
                cos_similarity = torch.matmul(key_states_norm, anchor_key.transpose(-1, -2))
                eviction_score = -cos_similarity  # (batch, heads, seq_len, 1)
                score = eviction_score.squeeze(-1).flatten()  # Flatten to 1D
                kv.flattened_scores[layer_idx].append(score)
            # Initialize KV meta info
            if len(key_states.shape) == 4:
                kv.info = {}
                _, heads_kv, seq_len, _ = key_states.shape
                cu_len_k = torch.arange(heads_kv + 1, dtype=torch.int32, device=self.device) * seq_len
                len_k = torch.full((heads_kv,), seq_len, dtype=torch.int32, device=self.device)
                kv.info["cu_len_k"] = [cu_len_k for _ in range(kv.n_layers)]
                kv.info["len_k"] = [len_k for _ in range(kv.n_layers)]

            kv.get_score = False
        else:
            raise NotImplementedError(f"Scoring method {self.scoring_method} is not implemented")
        
        assert not kv.get_score, "kv.get_score should be False"

    @torch.inference_mode()
    def generate(
        self,
        query: Union[str, torch.Tensor],
        kv: Optional[Union[EvictCache]] = None,
        update_cache: bool = False,
    ) -> str:
        """ Obtain a model response to the query
            In default, we evict KV of query and generated answer after the generation by kv.slice (for multi-query evaluation).
            Set update_cache = True to enable multi-turn generation.
        """
        kv = self._init_kv(kv=kv)
        seen_token_prev = kv._seen_tokens
        input_ids = query

        if type(query) == str:
            input_ids = self.encode(query)
        if kv.prefill_ids is not None:
            input_ids = torch.cat([kv.prefill_ids, input_ids], dim=1)

        output = self.model.generate(input_ids, past_key_values=kv, **self.gen_kwargs)
        a_ids = output[:, len(input_ids[0]):-1]  # parse response
        num_generated_tokens = a_ids.shape[1]
        a = self.decode(a_ids)

        if not update_cache:
            kv.slice(seen_token_prev)
        else:
            kv.prefill_ids = torch.cat([input_ids, a_ids], dim=1)
        return a, num_generated_tokens

