# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
from openai import OpenAI
from collections import defaultdict
import statistics
import os
import argparse
import glob
from typing import Dict, List, Tuple
from tqdm import tqdm

# OpenAI API configuration
# Get API key from environment variable or set directly
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def gpt_score(question: str, prediction: str, ground_truth: str) -> float:
    """
    Use ChatGPT API to compare prediction and ground_truth and return a score between 0-1
    """
    prompt = f"""Given the question and its ground truth answer, evaluate the correctness of the model's prediction.

Question: {question}
Ground truth: {ground_truth}
Model's prediction: {prediction}

Assign a score between 0 and 1, where 0 indicates the model's prediction is completely incorrect, and 1 indicates the model's prediction is completely correct.
Output in following JSON format:
{{
    "score": <score>,
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # Parse JSON response
        content = response.choices[0].message.content
        import re
        score_match = re.search(r'"score":\s*([0-9]*\.?[0-9]+)', content)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))  # Limit to 0-1 range
        else:
            print(f"Warning: Could not parse score from response: {content}")
            return 0.0
            
    except Exception as e:
        print(f"Error getting GPT score: {e}")
        print("Continuing with score 0.0...")
        return 0.0

def evaluate_json_results(json_file_path: str) -> Dict:
    """
    Read JSON file and score each element's prediction and ground_truth using ChatGPT API
    """
    # Read JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dictionary for storing results
    type_scores = defaultdict(list)
    all_scores = []
    
    print(f"Processing {len(data['individual_results'])} results...")
    
    # Process each element
    for key, item in tqdm(data['individual_results'].items(), desc=f"Evaluating {os.path.basename(json_file_path)}"):
        question = item['question']
        prediction = item['prediction']
        ground_truth = item['ground_truth']
        item_type = item['type']
        
        # Calculate score using ChatGPT API
        score = gpt_score(question, prediction, ground_truth)
        
        # Store results
        type_scores[item_type].append(score)
        all_scores.append(score)
        
        # Add GPT score to the original JSON data
        data['individual_results'][key]['gpt_score'] = score
    
    # Calculate statistics
    type_averages = {}
    for type_id, scores in type_scores.items():
        type_averages[type_id] = {
            'count': len(scores),
            'average': statistics.mean(scores),
            'median': statistics.median(scores),
            'min': min(scores),
            'max': max(scores)
        }
    
    overall_stats = {
        'count': len(all_scores),
        'average': statistics.mean(all_scores),
        'median': statistics.median(all_scores),
        'min': min(all_scores),
        'max': max(all_scores)
    }
    
    # Add evaluation results to the JSON data
    data['evaluation_results'] = {
        'type_averages': type_averages,
        'overall_stats': overall_stats
    }
    
    return {
        'type_averages': type_averages,
        'overall_stats': overall_stats,
        'data': data  # Return the updated data with GPT scores
    }

def print_results(evaluation_results: Dict, filename: str):
    """
    Print results in a nice format
    """
    print("\n" + "="*50)
    print(f"EVALUATION RESULTS: {filename}")
    print("="*50)
    
    # Print type averages
    print("\nType averages:")
    print("-" * 30)
    for type_id, stats in evaluation_results['type_averages'].items():
        print(f"Type {type_id}:")
        print(f"  Count: {stats['count']}")
        print(f"  Average: {stats['average']:.3f}")
        # print(f"  Median: {stats['median']:.3f}")
        # print(f"  Min: {stats['min']:.3f}")
        # print(f"  Max: {stats['max']:.3f}")
        print()
    
    # Print overall statistics
    overall = evaluation_results['overall_stats']
    print("Overall statistics:")
    print("-" * 30)
    print(f"Total Count: {overall['count']}")
    print(f"Overall Average: {overall['average']:.3f}")
    # print(f"Overall Median: {overall['median']:.3f}")
    # print(f"Overall Min: {overall['min']:.3f}")
    # print(f"Overall Max: {overall['max']:.3f}")

def process_directory(directory_path: str):
    """
    Process all JSON files in the given directory
    """
    # Find all JSON files in the directory
    json_pattern = os.path.join(directory_path, "**/*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    
    # Filter out files that already have gpt_score
    files_to_process = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if any item already has gpt_score
                has_gpt_score = any('gpt_score' in item for item in data.get('individual_results', {}).values())
                if not has_gpt_score:
                    files_to_process.append(json_file)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    print(f"Found {len(files_to_process)} JSON files to process")
    
    # Process each JSON file
    for json_file in files_to_process:
        try:
            print(f"\nProcessing: {json_file}")
            results = evaluate_json_results(json_file)
            
            # Print results
            print_results(results, os.path.basename(json_file))
            
            # Save updated JSON file with GPT scores
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results['data'], f, indent=2, ensure_ascii=False)
            print(f"Updated JSON saved to: {json_file}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate JSON results using GPT scoring')
    parser.add_argument('--directory', type=str, help='Directory containing JSON files to evaluate')
    args = parser.parse_args()
    
    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist!")
        return
    
    # Process all JSON files in the directory
    process_directory(args.directory)

if __name__ == "__main__":
    main() 