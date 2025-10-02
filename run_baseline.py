# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import os
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np

from args import args
from data import load_dataset_all
from model import LongConvQAModel
from utils import f1_score
from utils.cluster import ClusterManager
from utils.func import TimeStamp

if __name__ == "__main__":
    
    # ========================================================= #
    # Load Model
    # ========================================================= #
    model = LongConvQAModel(args.model, dtype=args.dtype, evict_level=args.level, scoring_method=args.scoring_method)
    dataset, _ = load_dataset_all(args.data, model.tokenizer, target_length=args.target_length)  # list of data
    
    # Store results for each question
    all_results = {}
    type_scores = defaultdict(list)
    total_scores = []
    
    # ========================================================= #
    # Online Evaluation (LLM Inference)
    # ========================================================= #
    print(">>> Running Online Evaluation...")
    for data_idx, data in enumerate(dataset):

        ctx_ids = model.encode(data['context'])
        kv = model.prefill_memory_constrained(ctx_ids, prefill_chunk_size=args.prefill_chunk_size, \
            score_path=args.score_path, kv_budget=args.kv_budget, power=args.power)

        if args.verbose and data_idx == 0:
            kv_budget = kv._budget()
            print(f">>> Avg Budget: {kv_budget.mean().item()} | Max Budget: {kv_budget.max().item()} | Min Budget: {kv_budget.min().item()} | Memory Usage: {kv._mem()} GB")
        
        for question_idx in tqdm(
            range(len(data['question'])),
            desc=f"Answering questions for conv {data_idx}",
            disable=not args.verbose,
            leave=False  # Progress bar will be removed after finishing
        ):
            question = data['question'][question_idx]
            answer = data['answers'][question_idx]
            question_type = data['task_types'][question_idx]
            
            pred, num_generated_tokens = model.generate(model.apply_template(question), kv=kv)
            score = f1_score(pred, answer)
             
            # Store result for this question
            result_key = f"conv_{data_idx}_q_{question_idx}"
            all_results[result_key] = {
                'conv_idx': data_idx,
                'question_idx': question_idx,
                'question': question,
                'prediction': pred,
                'ground_truth': answer,
                'f1_score': score,
                'type': question_type
            }
            
            # Track scores for type-wise and overall averages
            type_scores[question_type].append(score)
            total_scores.append(score)
        
    # Calculate average scores
    type_averages = {}
    for type_name, scores in type_scores.items():
        type_averages[type_name] = {
            'average_f1': sum(scores) / len(scores),
            'count': len(scores)
        }
    
    overall_average = sum(total_scores) / len(total_scores) if total_scores else 0
    
    # Combine all results
    final_results = {
        'individual_results': all_results,
        'type_averages': type_averages,
        'overall_average': {
            'average_f1': overall_average,
            'total_count': len(total_scores)
        }
    }
    
    # Save to JSON file
    output_file_name = f"{args.scoring_method}_{args.model.split('/')[-1]}_{args.data}.json"
    output_dir = f"results/{args.exp_name}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, output_file_name)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Overall Average F1: {overall_average:.4f} ({len(total_scores)} questions)")
    print(f"Type-wise averages:")
    for type_name, stats in type_averages.items():
        print(f"  {type_name}: {stats['average_f1']:.4f} ({stats['count']} questions)")
    print(f"Results saved to: {output_file}")
    print("Finished.")
