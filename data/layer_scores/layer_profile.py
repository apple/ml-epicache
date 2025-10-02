# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
import numpy as np
import argparse
import os
import json
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")


def _create_block_only_mask(batch_size, seq_length, device, window_size, sink_size, overlap_size, num_blocks, total_length):
    """Create attention mask for block-wise processing without memory tokens"""
    # Initialize mask: attend to nothing
    attention_mask = torch.zeros((batch_size, seq_length, seq_length), device=device)
    
    # Add sink tokens: tokens after sink_size can attend to all sink tokens
    sink_size = min(sink_size, seq_length)
    if seq_length > sink_size:
        attention_mask[:, sink_size:, :sink_size] = 1
    
    for b in range(batch_size):
        for block_idx in range(num_blocks):
            # Calculate current block boundaries
            window_start = block_idx * window_size
            window_end = min(window_start + window_size, total_length)
            
            # Rule 1: Causal attention within current block
            for i in range(window_start, window_end):
                attention_mask[b, i, window_start:i+1] = 1
            
            # Rule 2: Attend to overlap_size tokens from previous block
            if block_idx > 0 and overlap_size > 0:
                prev_window_start = (block_idx - 1) * window_size
                prev_window_end = min(prev_window_start + window_size, total_length)
                overlap_start = max(prev_window_start, prev_window_end - overlap_size)
                
                for i in range(window_start, window_end):
                    attention_mask[b, i, overlap_start:prev_window_end] = 1
    
    return attention_mask


def extract_key_caches_from_forward(model, input_ids, attention_mask=None, use_cache=True):
    """Perform forward pass and extract only key caches from all layers"""
    batch_size, seq_len = input_ids.shape
    
    # Create a DynamicCache to store KV values
    past_key_values = DynamicCache()
    
    # Prepare causal mask mapping if attention_mask is provided
    causal_mask_mapping = None
    if attention_mask is not None:
        inverted_mask = 1.0 - attention_mask
        sdpa_mask = inverted_mask * -10000.0
        causal_mask_mapping = sdpa_mask.to(model.dtype)
    
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=causal_mask_mapping,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=False,  # We don't need hidden states
            return_dict=True
        )
    
    # Extract only key caches from the past_key_values
    key_caches = {}
    num_layers = len(model.model.layers)
    
    for layer_idx in range(num_layers):
        if hasattr(outputs.past_key_values, 'key_cache'):
            # For DynamicCache
            key_cache = outputs.past_key_values.key_cache[layer_idx]  # [batch, num_heads, seq_len, head_dim]
        else:
            # Fallback: try to access directly from past_key_values tuple
            layer_past = outputs.past_key_values[layer_idx]
            key_cache = layer_past[0]
            
        key_caches[layer_idx] = key_cache.clone()
    
    return key_caches


def compute_windowed_key_similarity(key_cache_full, key_cache_custom, window_size=100):
    """Compute windowed cosine similarity between two key caches"""
    num_layers = len(key_cache_full)
    layer_scores = []
    
    for layer_idx in range(num_layers):
        key_full = key_cache_full[layer_idx]  # [batch, num_heads, seq_len, head_dim]
        key_custom = key_cache_custom[layer_idx]
        
        batch_size, num_heads, seq_len, head_dim = key_full.shape
        
        # Calculate number of windows
        num_windows = (seq_len + window_size - 1) // window_size  # Ceiling division
        
        # Initialize similarity matrix [num_heads, num_windows]
        key_similarities = torch.zeros(num_heads, num_windows)
        
        for head_idx in range(num_heads):
            for window_idx in range(num_windows):
                # Calculate window boundaries
                start_idx = window_idx * window_size
                end_idx = min(start_idx + window_size, seq_len)
                
                # Extract windowed embeddings [window_len, head_dim]
                key_full_window = key_full[0, head_idx, start_idx:end_idx]
                key_custom_window = key_custom[0, head_idx, start_idx:end_idx]
                
                # Flatten windows for similarity computation
                key_full_flat = key_full_window.flatten()
                key_custom_flat = key_custom_window.flatten()
                
                # Cosine similarity
                key_sim = torch.nn.functional.cosine_similarity(
                    key_full_flat.unsqueeze(0), key_custom_flat.unsqueeze(0)
                ).item()
                
                key_similarities[head_idx, window_idx] = key_sim
        
        # Calculate layer average (average across all heads and windows)
        layer_avg_score = key_similarities.mean().item()
        layer_scores.append(layer_avg_score)
    
    return layer_scores


def analyze_sample_key_similarity(model, input_ids, attention_window_size, sink_size, overlap_ratio, similarity_window_size):
    """Analyze key cache similarities for a single sample"""
    seq_length = input_ids.shape[1]
    
    # Create custom attention mask
    num_blocks = seq_length // attention_window_size + 1
    overlap_size = int(attention_window_size * overlap_ratio)
    
    custom_attention_mask = _create_block_only_mask(
        batch_size=1,
        seq_length=seq_length,
        device=model.device,
        window_size=attention_window_size,
        sink_size=sink_size,
        overlap_size=overlap_size,
        num_blocks=num_blocks,
        total_length=seq_length,
    )
    
    # Forward pass with full attention (baseline)
    key_cache_full = extract_key_caches_from_forward(model, input_ids, attention_mask=None)
    
    # Forward pass with custom attention mask
    key_cache_custom = extract_key_caches_from_forward(model, input_ids, attention_mask=custom_attention_mask)
    
    # Compute key cache similarities
    layer_scores = compute_windowed_key_similarity(
        key_cache_full, key_cache_custom, 
        window_size=similarity_window_size
    )
    
    return layer_scores


def profile_single_sample(model_path, input_file, sample_idx=0, 
                         attention_window_size=4096, similarity_window_size=100, 
                         sink_size=128, overlap_ratio=0.5, save_dir="results/layer_scores"):
    """Profile key cache similarities for a single sample from preprocessed .pt file"""
    
    # Load model
    print(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.eval()
    
    # Load preprocessed dataset
    print(f"Loading sample from .pt file: {input_file}")
    dataset_list = torch.load(input_file)
    
    if not isinstance(dataset_list, list):
        raise ValueError("Expected dataset to be a list of samples")
    
    if sample_idx >= len(dataset_list):
        raise ValueError(f"Sample index {sample_idx} out of range. Dataset has {len(dataset_list)} samples")
    
    sample = dataset_list[sample_idx]
    
    if not isinstance(sample, dict) or "input_ids" not in sample:
        raise ValueError("Sample should be a dict with 'input_ids' key")
    
    # Get input_ids tensor and move to device
    input_ids = sample["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    
    # Ensure proper shape [batch_size, seq_len]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    input_ids = input_ids.to(model.device)
    sample_name = f"{os.path.basename(input_file).replace('.pt', '')}_sample{sample_idx}"
        
    seq_length = input_ids.shape[1]
    print(f"Sample sequence length: {seq_length}")
    
    # Analyze sample and get layer scores
    layer_scores = analyze_sample_key_similarity(
        model, input_ids, attention_window_size, 
        sink_size, overlap_ratio, similarity_window_size
    )
    
    print(f"Average key similarity: {np.mean(layer_scores):.4f}")
    
    # Save results in JSON format
    os.makedirs(save_dir, exist_ok=True)
    model_name = model_path.split('/')[-1]
    save_path = os.path.join(save_dir, f"{sample_name}_layer_scores.json")
    
    # Prepare final results
    final_results = {
        "model_path": model_path,
        "input_file": input_file,
        "sample_idx": sample_idx,
        "sequence_length": seq_length,
        "config": {
            "attention_window_size": attention_window_size,
            "similarity_window_size": similarity_window_size,
            "sink_size": sink_size,
            "overlap_ratio": overlap_ratio,
        },
        "combined_scores": layer_scores
    }
    
    with open(save_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")
    print(f"Number of layer scores: {len(layer_scores)}")
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile key cache similarities for a single sample from preprocessed .pt file")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Path to the model")
    parser.add_argument("--input_file", type=str, default="data/booksum/booksum_Qwen2.5-7B-Instruct.pt",
                       help="Path to preprocessed .pt file containing list of samples")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Index of sample to analyze from the dataset")
    parser.add_argument("--attention_window_size", type=int, default=4096, 
                       help="Block size for custom attention mask")
    parser.add_argument("--similarity_window_size", type=int, default=100, 
                       help="Window size for similarity computation")
    parser.add_argument("--sink_size", type=int, default=128, 
                       help="Number of sink tokens")
    parser.add_argument("--overlap_ratio", type=float, default=0.5, 
                       help="Overlap ratio between blocks")
    parser.add_argument("--save_dir", type=str, default="data/layer_scores", 
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Run analysis
    results = profile_single_sample(
        model_path=args.model_path,
        input_file=args.input_file,
        sample_idx=args.sample_idx,
        attention_window_size=args.attention_window_size,
        similarity_window_size=args.similarity_window_size,
        sink_size=args.sink_size,
        overlap_ratio=args.overlap_ratio,
        save_dir=args.save_dir
    )
    
    print("\nSingle sample key cache similarity analysis completed!")