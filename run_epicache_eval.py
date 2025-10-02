# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import os
from collections import defaultdict
from tqdm import tqdm

from args import args
from data import load_dataset_all
from model import LongConvQAModel
from utils import f1_score
from utils.cluster import ClusterManager

if __name__ == "__main__":
    
    # ========================================================= #
    # Load Model
    # ========================================================= #

    model = LongConvQAModel(args.model, dtype=args.dtype, evict_level=args.level, scoring_method=args.scoring_method)

    # Initialize ClusterManager
    cluster_manager = ClusterManager(
        embedding_type=args.embedding_type,
        n_clusters=args.n_cluster,
        conv_window=args.conv_window,
        medoid_number=args.n_medoid,
        verbose=args.verbose
    )

    # ========================================================= #
    # Offline Clustering (Run Once)
    # ========================================================= #
    print(">>> Running Offline Clustering...")

    if args.data == "longmemeval":
        data_path = f"data/longmemeval/custom_lme_{args.target_length}_50.json"
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        dataset = dataset['conversations']
    else:
        data_path = "./data/locomo/locomo10.json" if args.data == "locomo" else "./data/realtalk/realtalk10.json"
        with open(data_path, 'r') as f:
            dataset = json.load(f)

    for conv_idx in tqdm(range(len(dataset)), desc=f"Conversation Clustering...", disable=not args.verbose, leave=False):
        
        conversation = dataset[conv_idx]

        if args.data == "longmemeval":
            conversation_data = conversation
            conversation_windows = cluster_manager.extract_conversation_windows_longmemeval(conversation_data)
        else:
            conversation_data = conversation['conversation'] if args.data == "locomo" else conversation
            conversation_windows = cluster_manager.extract_conversation_windows(conversation_data)

        embedded_windows = cluster_manager.embed_conversations(conversation_windows, model=model)
        clustering_results = cluster_manager.cluster_conversations(embedded_windows)
        
        # Save results for this conversation
        conversation_result = {
            'conv_idx': conv_idx,
            'embedded_windows': embedded_windows,
            'clustering_results': clustering_results,
            'n_windows': len(embedded_windows)
        }
        cluster_manager.all_conversation_results.append(conversation_result)
        
        for cluster_id in range(cluster_manager.n_clusters):
            cluster_info = clustering_results['cluster_results'][cluster_id]
            
    dataset, _ = load_dataset_all(args.data, model.tokenizer, target_length=args.target_length)  # list of data

    print(">>> Mapping questions to clusters...")
    # Create question-to-cluster pre-mappings for evaluation per-cluster
    cluster_manager.question_cluster_mappings = cluster_manager.create_question_cluster_mappings(dataset, model)
    
    # Store results for each question
    all_results = {}
    type_scores = defaultdict(list)
    total_scores = []
    
    # ========================================================= #
    # Online Evaluation (LLM Inference)
    # ========================================================= #
    print(">>> Running Online Evaluation...")
    for data_idx, data in enumerate(dataset):
        
        cluster_mappings = defaultdict(list)
        for mapping in cluster_manager.question_cluster_mappings[data_idx]:
            cluster_idx = mapping['closest_cluster']
            cluster_mappings[cluster_idx].append(mapping)
        
        # Then process each cluster
        for cluster_idx in tqdm(range(cluster_manager.n_clusters), desc=f"Processing clusters for conv {data_idx}", disable=not args.verbose, leave=False):
            # Get clustering results for this conversation
            clustering_results = cluster_manager.all_conversation_results[data_idx]['clustering_results']
            
            # Create cluster prompt
            combined_text = cluster_manager.make_cluster_prompt(clustering_results['cluster_results'][cluster_idx]['windows'])
            
            # Prefill
            model.cluster_ids = model.encode(combined_text)
            ctx_ids = model.encode(data['context'])
            kv = model.prefill_memory_constrained(ctx_ids, prefill_chunk_size=args.prefill_chunk_size, \
                score_path=args.score_path, kv_budget=args.kv_budget, power=args.power)

            if args.verbose and cluster_idx == 0 and data_idx == 0:
                kv_budget = kv._budget()
                print(f">>> Avg Budget: {kv_budget.mean().item()} | Max Budget: {kv_budget.max().item()} | Min Budget: {kv_budget.min().item()} | Memory Usage: {kv._mem()} GB")
            
            # Evaluate only cluster closed question
            for mapping in cluster_mappings[cluster_idx]:
                question_idx = mapping['question_idx']
                question = mapping['question']
                answer = mapping['answer']
                type = mapping['type']

                if args.data == "realtalk" and args.model == "Qwen/Qwen2.5-3B-Instruct":
                    question = "\n".join([q for i, q in enumerate(question.split("\n")) if i != 1])

                pred, _ = model.generate(model.apply_template(question), kv=kv)
                score = f1_score(pred, answer)
                
                # Store result for this question
                result_key = f"conv_{data_idx}_cluster_{cluster_idx}_q_{question_idx}"
                all_results[result_key] = {
                    'conv_idx': data_idx,
                    'cluster_idx': cluster_idx,
                    'question_idx': question_idx,
                    'question': question,
                    'prediction': pred,
                    'ground_truth': answer,
                    'f1_score': score,
                    'type': type
                }
                
                # Track scores for type-wise and overall averages
                type_scores[type].append(score)
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
    output_file_name = f"cluster_{args.model.split('/')[-1]}_{args.data}.json"
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
