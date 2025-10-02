#!/usr/bin/env python3

# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import os
import torch
import argparse
from typing import List, Dict, Any


# Task type mapping from string to integer (similar to LocoMo categories)
TASK_TYPE_MAPPING = {
    'single_hop': 0,
    'two_hop': 1, 
    'multi_session_synthesis': 2,
    'temp_reasoning_explicit': 3,
    'temp_reasoning_implicit': 4,
    'knowledge_update': 5,
    'implicit_preference': 6,
    'implicit_preference_v2': 7
}

# QA prompt template (similar to LocoMo format)
QA_PROMPT = """
Based on the above conversations, write a short answer for the following question in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

Question: {}
"""

def build_conversation_context(conversation_timeline: List[Dict[str, Any]]) -> str:
    """
    Build the full conversation context from LongMemEval conversation timeline.
    Similar to LocoMo's conversation building but adapted for our format.
    """
    context_parts = []
    context_parts.append("Below is a conversation history between a user and an assistant. The conversation takes place over multiple sessions and the timestamp of each session is provided.\n\n")
    
    for session in conversation_timeline:
        # Add timestamp
        timestamp = session['timestamp']
        session_id = session['session_id']
        context_parts.append(f"TIMESTAMP: {timestamp}")
        context_parts.append(f"SESSION: {session_id}")
        context_parts.append("CONVERSATION:")
        
        # Add conversation turns
        for turn in session['session']:
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                context_parts.append(f"User: {content}")
            elif role == 'assistant':
                context_parts.append(f"Assistant: {content}")
        
        context_parts.append("")  # Empty line between sessions
    
    return "\n".join(context_parts)

def convert_longmemeval_to_scbench(input_path: str, output_path: str) -> None:
    """
    Convert LongMemEval custom dataset to SCBench/LocoMo format.
    """
    print(f"Loading LongMemEval data from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        longmemeval_data = json.load(f)
    
    scbench_data = []
    total_conversations = len(longmemeval_data['conversations'])
    total_qa_pairs = 0
    
    for conv_idx, conversation in enumerate(longmemeval_data['conversations']):
        print(f"Processing conversation {conv_idx + 1}/{total_conversations}: {conversation['conversation_id']}")
        
        # Build conversation context
        conversation_context = build_conversation_context(conversation['conversation_timeline'])
        
        # Prepare prompts, answers, and task types
        prompts = [conversation_context]
        ground_truth = []
        task_types = []
        
        # Process QA pairs
        for qa_pair in conversation['qa_pairs']:
            question = qa_pair['question']
            answer = qa_pair['answer']
            question_type = qa_pair['question_type']
            
            # Convert answer to string if needed
            if isinstance(answer, (int, float)):
                answer = str(answer)
            elif not isinstance(answer, str):
                answer = str(answer)
            
            # Format question with prompt template
            formatted_question = QA_PROMPT.format(question)
            
            prompts.append(formatted_question)
            ground_truth.append(answer)
            
            # Map task type to integer
            task_type_int = TASK_TYPE_MAPPING.get(question_type, 0)
            task_types.append(task_type_int)
            
            total_qa_pairs += 1
        
        # Create SCBench format entry
        scbench_entry = {
            'prompts': prompts,
            'ground_truth': ground_truth,
            'task_types': task_types,
            'options': [],  # No multiple choice options
            'conversation_id': conversation['conversation_id'],  # Keep original ID for reference
            'total_tokens': conversation['total_tokens'],
            'metadata': conversation.get('metadata', {})
        }
        
        scbench_data.append(scbench_entry)
    
    # Save as .pt file (PyTorch format like other SCBench data)
    print(f"Saving converted data to: {output_path}")
    torch.save(scbench_data, output_path)
    
    print(f"Conversion complete!")
    print(f"- Total conversations: {len(scbench_data)}")
    print(f"- Total QA pairs: {total_qa_pairs}")
    print(f"- Average QA pairs per conversation: {total_qa_pairs / len(scbench_data):.1f}")
    
    # Print task type distribution
    all_task_types = []
    for item in scbench_data:
        all_task_types.extend(item['task_types'])
    
    task_type_counts = {}
    for task_type in all_task_types:
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    print("\nTask type distribution:")
    reverse_mapping = {v: k for k, v in TASK_TYPE_MAPPING.items()}
    for task_type, count in sorted(task_type_counts.items()):
        task_name = reverse_mapping.get(task_type, f"unknown_{task_type}")
        percentage = (count / total_qa_pairs) * 100
        print(f"  Type {task_type} ({task_name}): {count} questions ({percentage:.1f}%)")
    
    # Print token statistics
    token_counts = [item['total_tokens'] for item in scbench_data]
    print(f"\nToken statistics:")
    print(f"  Min tokens: {min(token_counts):,}")
    print(f"  Max tokens: {max(token_counts):,}")
    print(f"  Mean tokens: {sum(token_counts) / len(token_counts):,.1f}")

def validate_converted_data(converted_path: str) -> None:
    """
    Validate the converted data format and show sample.
    """
    print(f"\nValidating converted data from: {converted_path}")
    
    try:
        test_data = torch.load(converted_path)
        print(f"✓ Successfully loaded {len(test_data)} conversations")
        
        if len(test_data) > 0:
            first_conv = test_data[0]
            
            # Check required keys
            required_keys = ['prompts', 'ground_truth', 'task_types', 'options']
            for key in required_keys:
                if key not in first_conv:
                    print(f"✗ Missing required key: {key}")
                    return
                else:
                    print(f"✓ Found key: {key}")
            
            # Check data consistency
            num_prompts = len(first_conv['prompts'])
            num_answers = len(first_conv['ground_truth'])
            num_task_types = len(first_conv['task_types'])
            
            print(f"\nFirst conversation structure:")
            print(f"  - Prompts: {num_prompts} (1 context + {num_prompts-1} questions)")
            print(f"  - Answers: {num_answers}")
            print(f"  - Task types: {num_task_types}")
            print(f"  - Options: {len(first_conv['options'])}")
            
            if num_answers != num_task_types:
                print(f"✗ Mismatch: {num_answers} answers vs {num_task_types} task types")
                return
            
            if num_prompts - 1 != num_answers:
                print(f"✗ Mismatch: {num_prompts-1} questions vs {num_answers} answers")
                return
            
            print(f"✓ Data structure is consistent")
            
            # Show sample content
            print(f"\nSample content:")
            context_length = len(first_conv['prompts'][0])
            print(f"  Context length: {context_length:,} characters")
            print(f"  Context preview: {first_conv['prompts'][0][:200]}...")
            
            if len(first_conv['prompts']) > 1:
                print(f"  First question: {first_conv['prompts'][1][:200]}...")
                print(f"  First answer: {first_conv['ground_truth'][0]}")
                print(f"  First task type: {first_conv['task_types'][0]}")
            
            print(f"✓ Validation successful!")
            
    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")

def create_load_function(output_path: str) -> None:
    """
    Create a load function that can be integrated into the evaluation framework.
    """
    load_function_code = f'''
def load_longmemeval(path="{output_path}"):
    """
    Load LongMemEval dataset from preprocessed .pt file
    Compatible with the evaluation framework's data loading interface.
    """
    import torch
    import os
    
    dataset = []
    samples = torch.load(path)

    for data in samples:
        d = {{}}
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

        # Add metadata
        if "conversation_id" in data:
            d["conversation_id"] = data["conversation_id"]
        if "total_tokens" in data:
            d["total_tokens"] = data["total_tokens"]

        dataset.append(d)

    return dataset
'''
    
    load_function_path = output_path.replace('.pt', '_load.py')
    with open(load_function_path, 'w', encoding='utf-8') as f:
        f.write(load_function_code)
    
    print(f"Created load function at: {load_function_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert LongMemEval custom dataset to SCBench/LocoMo format')
    parser.add_argument('--input_path', type=str, help='Path to LongMemEval custom dataset JSON file')
    parser.add_argument('--output_path', type=str, default='longmemeval_preprocessed.pt',
                       help='Output path for converted .pt file')
    parser.add_argument('--validate', action='store_true', 
                       help='Validate the converted data after processing')
    parser.add_argument('--create_load_func', action='store_true',
                       help='Create a load function for integration with evaluation framework')
    
    args = parser.parse_args()
    
    # Convert the dataset
    convert_longmemeval_to_scbench(args.input_path, args.output_path)
    
    # Validate if requested
    if args.validate:
        validate_converted_data(args.output_path)
    
    # Create load function if requested
    if args.create_load_func:
        create_load_function(args.output_path)
    
    print(f"\nConversion pipeline completed!")
    print(f"Use the converted data in your evaluation framework by loading: {args.output_path}")


if __name__ == "__main__":
    main() 