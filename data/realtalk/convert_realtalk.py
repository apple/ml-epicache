#!/usr/bin/env python3

# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import os
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer
CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation.\n\n"

QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase for the following question.
If the question is about a date, try to infer the approximate date (e.g., "In the 1800s", "Before Jan 2021", etc.).

Question: {}
Answer:
"""



def build_conversation_context(conversation_data: Dict[str, Any]) -> str:
    """
    Build the full conversation context from RealTalk conversation data.
    Similar to get_input_context function from qa_tool.py
    """
    # Get speaker names from the name field
    speaker_1 = conversation_data.get('name', {}).get('speaker_1', 'Speaker 1')
    speaker_2 = conversation_data.get('name', {}).get('speaker_2', 'Speaker 2')
    
    # Start with conversation prompt
    context = CONV_START_PROMPT.format(speaker_1, speaker_2)
    
    # Get all session numbers
    session_keys = [k for k in conversation_data.keys() if k.startswith('session_') and not k.endswith('_date_time')]
    session_numbers = []
    for key in session_keys:
        try:
            session_num = int(key.split('_')[1])
            session_numbers.append(session_num)
        except (ValueError, IndexError):
            continue
    
    session_numbers.sort()
    
    # Build conversation text
    conversation_text = ""
    for session_num in session_numbers:
        session_key = f'session_{session_num}'
        datetime_key = f'session_{session_num}_date_time'
        
        if session_key in conversation_data and conversation_data[session_key]:
            # Add date/time info
            if datetime_key in conversation_data:
                conversation_text += f"DATE: {conversation_data[datetime_key]}\n"
            
            conversation_text += "CONVERSATION:\n"
            
            # Add each dialog turn
            for dialog in conversation_data[session_key]:
                speaker = dialog.get('speaker', 'Unknown')
                text = dialog.get('clean_text', dialog.get('text', ''))
                
                turn = f'{speaker} said, "{text}"'
                
                # Handle blip_caption if present (from qa_tool.py)
                if "blip_caption" in dialog:
                    turn += f'\n{speaker} shared, an image of "{dialog["blip_caption"]}".'
                
                turn += "\n"
                conversation_text += turn
            
            conversation_text += "\n"
    
    full_context = context + conversation_text
    return full_context

def convert_realtalk_to_scbench(realtalk_path: str, output_path: str) -> None:
    """
    Convert RealTalk format to SCBench format
    """
    print(f"Loading RealTalk data from: {realtalk_path}")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    with open(realtalk_path, 'r') as f:
        realtalk_data = json.load(f)
    
    scbench_data = []
    total_skipped = 0
    total_conv_length = 0
    for conv_idx, conversation in enumerate(realtalk_data):
        print(f"Processing conversation {conv_idx + 1}/{len(realtalk_data)}")
        
        # Build conversation context
        conversation_context = build_conversation_context(conversation)
        total_conv_length += tokenizer(conversation_context, return_tensors="pt")['input_ids'].shape[1]
        # Get questions and answers
        qa_list = conversation['qa']
        
        # Group questions by conversation (each conversation becomes one SCBench item)
        # First prompt is the context, rest are questions
        prompts = [conversation_context]
        ground_truth = []
        task_types = []  # Add task_types tracking
        
        for qa_idx, qa in enumerate(qa_list):
            # Skip tasks without 'answer' key
            if 'answer' not in qa:
                total_skipped += 1
                continue
            
            question = qa['question']
            answer = qa['answer']
            category = qa.get('category', 0)  # Get task type (category)
            
            # Convert answer to string if it's not already
            if isinstance(answer, (int, float)):
                answer = str(answer)
            elif not isinstance(answer, str):
                answer = str(answer)
            
            # Format question with QA_PROMPT template
            formatted_question = QA_PROMPT.format(question)
            
            prompts.append(formatted_question)
            ground_truth.append(answer)
            task_types.append(category)  # Store task type
        
        # Only add conversation if it has valid QA pairs
        if len(ground_truth) > 0:
            # Create SCBench format entry
            scbench_entry = {
                'prompts': prompts,
                'ground_truth': ground_truth,
                'task_types': task_types,  # Add task_types field
                'options': []  # RealTalk doesn't have multiple choice options
            }
            
            scbench_data.append(scbench_entry)
    
    print(f"Average conversation length: {total_conv_length / len(scbench_data)}")
    # Save as .pt file (like other SCBench data)
    print(f"Saving converted data to: {output_path}")
    torch.save(scbench_data, output_path)
    
    print(f"Conversion complete!")
    print(f"- Total conversations: {len(scbench_data)}")
    print(f"- Total questions: {sum(len(item['ground_truth']) for item in scbench_data)}")
    print(f"- Skipped tasks without answers: {total_skipped}")
    
    # Print task type distribution
    all_task_types = []
    for item in scbench_data:
        all_task_types.extend(item['task_types'])
    
    task_type_counts = {}
    for task_type in all_task_types:
        task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
    
    print("\nTask type distribution:")
    for task_type, count in sorted(task_type_counts.items()):
        print(f"  Type {task_type}: {count} questions")

def main():
    # Input and output paths (relative to current directory)
    realtalk_path = "realtalk10.json"
    output_path = "realtalk_preprocessed.pt"
    
    # Convert the data
    convert_realtalk_to_scbench(realtalk_path, output_path)
    
    # Test loading the converted data
    print("\nTesting converted data...")
    test_data = torch.load(output_path)
    print(f"Loaded {len(test_data)} conversations")
    
    if len(test_data) > 0:
        first_conv = test_data[0]
        print(f"First conversation has {len(first_conv['prompts'])} prompts")
        print(f"First conversation has {len(first_conv['ground_truth'])} answers")
        print(f"First conversation has {len(first_conv['task_types'])} task types")
        print(f"Context length: {len(first_conv['prompts'][0])}")
        
        # Show first question as example
        if len(first_conv['prompts']) > 1:
            print(f"\nFirst question: {first_conv['prompts'][1][:200]}...")
            print(f"First answer: {first_conv['ground_truth'][0]}")
            print(f"First task type: {first_conv['task_types'][0]}")

if __name__ == "__main__":
    main() 