# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import os
import sys
import json
import random
from datetime import datetime, timedelta
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer
from collections import defaultdict
import argparse


class CustomDatasetGenerator:
    """
    Custom dataset generator for long memory evaluation.
    Creates conversations with 20-25K tokens containing multiple evidence sessions
    and corresponding QA pairs for comprehensive evaluation.
    """
    
    def __init__(self, 
                 data_dir='custom_history_data',
                 ref_model_name='meta-llama/Llama-3.1-8B-Instruct',
                 target_min_tokens=20000,
                 target_max_tokens=25000):
        
        self.data_dir = data_dir
        self.ref_model_name = ref_model_name
        self.target_min_tokens = target_min_tokens
        self.target_max_tokens = target_max_tokens
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        
        # Load all databases
        self._load_databases()
        
        # Task type configurations - based on actual 500 questions distribution
        self.task_type_weights = {
            'single_hop': 0.14,           # 70/500 = 14.0%
            'two_hop': 0.142,             # 71/500 = 14.2%
            'multi_session_synthesis': 0.124,  # 62/500 = 12.4%
            'temp_reasoning_explicit': 0.12,   # 60/500 = 12.0%
            'temp_reasoning_implicit': 0.146,  # 73/500 = 14.6%
            'knowledge_update': 0.156,         # 78/500 = 15.6%
            'implicit_preference_v2': 0.06,    # 30/500 = 6.0%
            'assistant_previnfo': 0.112        # 56/500 = 11.2%
        }
        
    def _load_databases(self):
        """Load all required database files."""
        print("Loading databases...")
        
        # Load attribute mappings
        attr_file = os.path.join(self.data_dir, '1_attr_bg/data_1_attr_bg.json')
        self.attribute_db = json.load(open(attr_file))
        
        # Create background_id to attribute_id mapping
        self.bgid_to_attribute = {}
        for attr_entry in self.attribute_db:
            for bg_entry in attr_entry['backgrounds']:
                self.bgid_to_attribute[bg_entry['background_id']] = bg_entry['attribute_id']
        
        # Load questions database
        question_file = os.path.join(self.data_dir, '2_questions/0822_all_500_questions_final_v2.json')
        self.question_db = json.load(open(question_file))
        
        # Create question_id to attribute_id mapping
        self.qid2attid = {
            x['question_id']: self.bgid_to_attribute.get(x['background_id'], x['background_id']) 
            for x in self.question_db
        }
        
        # Load session cache
        session_file = os.path.join(self.data_dir, '6_session_cache/data_6_session_cache.json')
        self.session_db = json.load(open(session_file))
        
        # Load filler sessions for assistant_previnfo
        filler_file = os.path.join(self.data_dir, '5_filler_sess/data_5_filler_sess.json')
        self.filler_db = json.load(open(filler_file))
        
        # Create session index (combine both)
        self.sess_idx = {x['session_id']: x for x in self.session_db}
        self.sess_idx.update({x['session_id']: x for x in self.filler_db})
        
        # Process user sessions (handle different session formats)
        self.user_sessions = self._process_user_sessions()
        
        # Group questions by task type
        self.questions_by_task = defaultdict(list)
        for question in self.question_db:
            if self._is_valid_question(question):
                self.questions_by_task[question['question_type']].append(question)
        
        print(f"Loaded {len(self.question_db)} questions across {len(self.questions_by_task)} task types")
        print(f"Task type distribution: {[(k, len(v)) for k, v in self.questions_by_task.items()]}")
    
    def _process_user_sessions(self):
        """Process user sessions to handle different formats (session, sessions, session_1/2, etc.)."""
        processed_sessions = []
        
        for session_entry in self.session_db:
            if 'session' in session_entry:
                processed_sessions.append(session_entry)
            elif 'sessions' in session_entry:
                for i, sess in enumerate(session_entry['sessions']):
                    new_entry = deepcopy(session_entry)
                    new_entry['session'] = sess
                    new_entry['session_id'] += f'_{i+1}'
                    processed_sessions.append(new_entry)
            elif 'session_1' in session_entry and 'session_2' in session_entry:
                for i, sess_key in enumerate(['session_1', 'session_2']):
                    new_entry = deepcopy(session_entry)
                    new_entry['session'] = session_entry[sess_key]
                    new_entry['session_id'] += f'_{i+1}'
                    processed_sessions.append(new_entry)
            elif 'old_session' in session_entry and 'new_session' in session_entry:
                for i, sess_key in enumerate(['old_session', 'new_session']):
                    new_entry = deepcopy(session_entry)
                    new_entry['session'] = session_entry[sess_key]
                    new_entry['session_id'] += f'_{i+1}'
                    processed_sessions.append(new_entry)
        
        return processed_sessions
    
    def _is_valid_question(self, question):
        """Check if a question has valid sessions for evidence."""
        if not question.get('sessions'):
            return False
        
        # Check if there are valid sessions with human_valid_label
        for sess_entry in question['sessions']:
            session_data = self.sess_idx.get(sess_entry['session_id'])
            if session_data and session_data.get('human_valid_label'):
                return True
        
        return False
    
    def _count_tokens(self, content):
        """Count tokens for given content."""
        if isinstance(content, str):
            return len(self.tokenizer.encode(content))
        elif isinstance(content, list):
            # Handle conversation sessions
            text = json.dumps(content)
            return len(self.tokenizer.encode(text))
        elif isinstance(content, dict):
            text = json.dumps(content)
            return len(self.tokenizer.encode(text))
        else:
            return 0
    
    def _sample_mixed_questions(self, max_questions=50):
        """Sample questions from different task types according to weights."""
        sampled_questions = []
        
        # Calculate target counts for each task type
        total_weight = sum(self.task_type_weights.values())
        
        for task_type, weight in self.task_type_weights.items():
            if task_type not in self.questions_by_task:
                print(f"Task type {task_type} not found in questions by task")
                continue
            else:
                print(f"Task type {task_type} found in questions by task")
                
            target_count = int((weight / total_weight) * max_questions)
            available_questions = self.questions_by_task[task_type]
            
            if target_count > len(available_questions):
                target_count = len(available_questions)
            
            sampled = random.sample(available_questions, target_count)
            sampled_questions.extend(sampled)
        
        # Shuffle to mix task types
        random.shuffle(sampled_questions)
        return sampled_questions
    
    def _get_evidence_sessions_for_question(self, question):
        """Get evidence sessions for a given question based on task type."""
        task_type = question['question_type']
        evidence_sessions = []
        
        if task_type in ['single_hop', 'implicit_preference', 'implicit_preference_v2']:
            # Single session with answer
            for sess_entry in question['sessions']:
                if sess_entry['style'] == 'neutral':
                    session_data = self.sess_idx.get(sess_entry['session_id'])
                    if session_data and session_data.get('human_valid_label'):
                        evidence_sessions.append({
                            'session_id': f"evidence_{session_data['session_id']}",
                            'session': session_data['session'],
                            'source_question_id': question['question_id'],
                            'task_type': task_type
                        })
                        break
        
        elif task_type == 'two_hop':
            # Two sessions with answers
            for sess_entry in question['sessions']:
                if sess_entry['style'] == 'neutral':
                    session_data = self.sess_idx.get(sess_entry['session_id'])
                    if session_data and session_data.get('human_valid_label'):
                        evidence_sessions.append({
                            'session_id': f"evidence_{session_data['session_id']}_1",
                            'session': session_data['session_1'],
                            'source_question_id': question['question_id'],
                            'task_type': task_type
                        })
                        evidence_sessions.append({
                            'session_id': f"evidence_{session_data['session_id']}_2",
                            'session': session_data['session_2'],
                            'source_question_id': question['question_id'],
                            'task_type': task_type
                        })
                        break
        
        elif task_type in ['multi_session_synthesis', 'temp_reasoning_explicit']:
            # Multiple sessions
            for sess_entry in question['sessions']:
                if sess_entry['style'] == 'neutral':
                    session_data = self.sess_idx.get(sess_entry['session_id'])
                    if session_data and session_data.get('human_valid_label'):
                        for i, sess in enumerate(session_data['sessions']):
                            evidence_sessions.append({
                                'session_id': f"evidence_{session_data['session_id']}_{i+1}",
                                'session': sess,
                                'source_question_id': question['question_id'],
                                'task_type': task_type
                            })
                        break
        
        elif task_type == 'knowledge_update':
            # Old and new sessions
            for sess_entry in question['sessions']:
                if sess_entry['style'] == 'neutral':
                    session_data = self.sess_idx.get(sess_entry['session_id'])
                    if session_data and session_data.get('human_valid_label'):
                        evidence_sessions.append({
                            'session_id': f"evidence_{session_data['session_id']}_old",
                            'session': session_data['session_old'],
                            'source_question_id': question['question_id'],
                            'task_type': task_type
                        })
                        evidence_sessions.append({
                            'session_id': f"evidence_{session_data['session_id']}_new",
                            'session': session_data['session_new'],
                            'source_question_id': question['question_id'],
                            'task_type': task_type
                        })
                        break
        
        elif task_type == 'temp_reasoning_implicit':
            # Sessions with specific dates
            for sess_entry in question['sessions']:
                if sess_entry['style'] == 'neutral':
                    session_data = self.sess_idx.get(sess_entry['session_id'])
                    if session_data and session_data.get('human_valid_label'):
                        for i, sess in enumerate(session_data['sessions']):
                            evidence_sessions.append({
                                'session_id': f"evidence_{session_data['session_id']}_{i+1}",
                                'session': sess,
                                'source_question_id': question['question_id'],
                                'task_type': task_type,
                                'fact_date': question['question_content']['facts'][i]['date'] if i < len(question['question_content'].get('facts', [])) else None
                            })
                        break
        
        elif task_type == 'assistant_previnfo':
            # Single session with answer (similar to single_hop)
            for sess_entry in question['sessions']:
                if sess_entry['style'] == 'neutral':
                    session_data = self.sess_idx.get(sess_entry['session_id'])
                    if session_data and session_data.get('human_valid_label'):
                        evidence_sessions.append({
                            'session_id': f"evidence_{session_data['session_id']}",
                            'session': session_data['session'],
                            'source_question_id': question['question_id'],
                            'task_type': task_type
                        })
                        break
        
        elif task_type == 'implicit_preference_v2':
            # Single session with answer (similar to single_hop)
            for sess_entry in question['sessions']:
                if sess_entry['style'] == 'neutral':
                    session_data = self.sess_idx.get(sess_entry['session_id'])
                    if session_data and session_data.get('human_valid_label'):
                        evidence_sessions.append({
                            'session_id': f"evidence_{session_data['session_id']}",
                            'session': session_data['session'],
                            'source_question_id': question['question_id'],
                            'task_type': task_type
                        })
                        break
        
        return evidence_sessions
    
    def _assign_timestamps(self, evidence_sessions, questions):
        """Assign timestamps to evidence sessions based on task type constraints."""
        # Group sessions by task type for appropriate timestamp assignment
        sessions_by_task = defaultdict(list)
        for session in evidence_sessions:
            sessions_by_task[session['task_type']].append(session)
        
        # Assign base dates for different task types
        base_date = datetime(2023, 5, 25)  # Fixed base date
        current_date = base_date
        
        for task_type, sessions in sessions_by_task.items():
            if task_type in ['single_hop', 'two_hop', 'multi_session_synthesis', 'implicit_preference', 'implicit_preference_v2', 'assistant_previnfo']:
                # Same day, different times
                dates = self._get_random_same_day_timestamps(len(sessions), current_date.strftime("%Y/%m/%d"))
                for i, session in enumerate(sessions):
                    session['timestamp'] = dates[i]
            
            elif task_type in ['temp_reasoning_explicit']:
                # Same day for sessions from same question
                question_sessions = defaultdict(list)
                for session in sessions:
                    question_sessions[session['source_question_id']].append(session)
                
                for q_sessions in question_sessions.values():
                    dates = self._get_random_same_day_timestamps(len(q_sessions), current_date.strftime("%Y/%m/%d"))
                    for i, session in enumerate(q_sessions):
                        session['timestamp'] = dates[i]
                    current_date += timedelta(days=1)
            
            elif task_type == 'temp_reasoning_implicit':
                # Use fact dates if available
                for session in sessions:
                    if session.get('fact_date'):
                        fact_date = self._format_date(session['fact_date'])
                        session['timestamp'] = self._get_random_same_day_timestamps(1, fact_date)[0]
                    else:
                        session['timestamp'] = self._get_random_same_day_timestamps(1, current_date.strftime("%Y/%m/%d"))[0]
            
            elif task_type == 'knowledge_update':
                # Maintain temporal order for old -> new
                question_sessions = defaultdict(list)
                for session in sessions:
                    question_sessions[session['source_question_id']].append(session)
                
                for q_sessions in question_sessions.values():
                    old_sessions = [s for s in q_sessions if 'old' in s['session_id']]
                    new_sessions = [s for s in q_sessions if 'new' in s['session_id']]
                    
                    old_date = current_date
                    new_date = current_date + timedelta(days=7)
                    
                    for session in old_sessions:
                        session['timestamp'] = self._get_random_same_day_timestamps(1, old_date.strftime("%Y/%m/%d"))[0]
                    for session in new_sessions:
                        session['timestamp'] = self._get_random_same_day_timestamps(1, new_date.strftime("%Y/%m/%d"))[0]
                    
                    current_date = new_date + timedelta(days=1)
        
        # Sort all sessions by timestamp
        def parse_timestamp(timestamp_str):
            try:
                # Handle format: "2023/05/25 (Thu) 14:30"
                date_part = timestamp_str.split(' (')[0]
                if len(date_part.split()) == 2:  # Has time
                    return datetime.strptime(date_part, "%Y/%m/%d %H:%M")
                else:  # Only date
                    return datetime.strptime(date_part, "%Y/%m/%d")
            except ValueError:
                # Fallback to a default timestamp
                return datetime(2023, 5, 25, 12, 0)
        
        # Debug: check which sessions don't have timestamp
        sessions_without_timestamp = [s for s in evidence_sessions if 'timestamp' not in s]
        if sessions_without_timestamp:
            print(f"Warning: {len(sessions_without_timestamp)} sessions without timestamp:")
            for s in sessions_without_timestamp[:3]:  # Show first 3
                print(f"  {s['task_type']}: {s['session_id']}")
        
        evidence_sessions.sort(key=lambda x: parse_timestamp(x['timestamp']))
        
        return evidence_sessions
    
    def _format_date(self, date_str):
        """Format date string to YYYY/MM/DD format."""
        parts = date_str.split('/')
        if len(parts) != 3:
            return date_str
        
        year, month, day = parts
        if len(month) == 1:
            month = '0' + month
        if len(day) == 1:
            day = '0' + day
        
        return f"{year}/{month}/{day}"
    
    def _get_random_same_day_timestamps(self, n, base_date=None):
        """Generate random timestamps for the same day."""
        if base_date is None:
            base_date = "2023/05/25"
        
        base_date = datetime.strptime(base_date, "%Y/%m/%d")
        
        random_times = []
        for _ in range(n):
            random_seconds = random.randint(0, 86399)  # Seconds in a day
            random_time = base_date + timedelta(seconds=random_seconds)
            random_times.append(random_time)
        
        random_times.sort()
        formatted_times = [time.strftime("%Y/%m/%d (%a) %H:%M") for time in random_times]
        return formatted_times
    
    def _create_qa_pair(self, question, evidence_sessions):
        """Create QA pair with metadata."""
        return {
            'question_id': question['question_id'],
            'question': question['question_content']['question'],
            'answer': question['question_content']['answer'],
            'question_type': question['question_type'],
            'evidence_session_ids': [s['session_id'] for s in evidence_sessions if s['source_question_id'] == question['question_id']],
            'decomp_facts': question['question_content'].get('decomp_facts', {}),
            'temporal_constraint': question['question_content'].get('temporal_constraint', {}),
            'facts': question['question_content'].get('facts', [])
        }
    
    def generate_conversation(self, conversation_id, max_attempts=100):
        """Generate a single conversation with target token length."""
        
        for attempt in range(max_attempts):
            # Sample questions
            candidate_questions = self._sample_mixed_questions(max_questions=60)
            
            evidence_sessions = []
            qa_pairs = []
            current_tokens = 0
            used_attributes = set()
            
            for question in candidate_questions:
                # Check attribute conflict
                question_attr = self.qid2attid.get(question['question_id'])
                if question_attr in used_attributes:
                    continue
                
                # Get evidence sessions for this question
                question_sessions = self._get_evidence_sessions_for_question(question)
                if not question_sessions:
                    continue
                
                # Calculate tokens for these sessions
                session_tokens = sum(self._count_tokens(s['session']) for s in question_sessions)
                
                # Check if we can add these sessions
                if current_tokens + session_tokens > self.target_max_tokens:
                    # If we haven't reached minimum, try with a smaller session
                    if current_tokens < self.target_min_tokens and session_tokens > 1000:
                        continue
                    else:
                        break
                
                # Add sessions and QA pair
                evidence_sessions.extend(question_sessions)
                qa_pairs.append(self._create_qa_pair(question, question_sessions))
                current_tokens += session_tokens
                used_attributes.add(question_attr)
                
                # Check if we've reached target
                if current_tokens >= self.target_min_tokens:
                    break
            
            # Check if we achieved target token range
            if self.target_min_tokens <= current_tokens <= self.target_max_tokens:
                # Assign timestamps and sort
                evidence_sessions = self._assign_timestamps(evidence_sessions, [qa['question_id'] for qa in qa_pairs])
                
                return {
                    'conversation_id': conversation_id,
                    'total_tokens': current_tokens,
                    'num_evidence_sessions': len(evidence_sessions),
                    'num_qa_pairs': len(qa_pairs),
                    'task_type_distribution': self._get_task_distribution(qa_pairs),
                    'conversation_timeline': [
                        {
                            'session_id': session['session_id'],
                            'timestamp': session['timestamp'],
                            'task_type': session['task_type'],
                            'session': session['session']
                        }
                        for session in evidence_sessions
                    ],
                    'qa_pairs': qa_pairs,
                    'metadata': {
                        'generation_date': datetime.now().isoformat(),
                        'target_token_range': f"{self.target_min_tokens}-{self.target_max_tokens}",
                        'model_tokenizer': self.ref_model_name
                    }
                }
        
        # If we couldn't generate a valid conversation after max_attempts
        return None
    
    def _get_task_distribution(self, qa_pairs):
        """Get task type distribution in QA pairs."""
        task_counts = defaultdict(int)
        for qa in qa_pairs:
            task_counts[qa['question_type']] += 1
        return dict(task_counts)
    
    def generate_dataset(self, num_conversations, output_file):
        """Generate complete dataset with multiple conversations."""
        print(f"Generating {num_conversations} conversations...")
        
        dataset = {
            'dataset_info': {
                'num_conversations': num_conversations,
                'target_tokens_per_conversation': f"{self.target_min_tokens}-{self.target_max_tokens}",
                'generation_date': datetime.now().isoformat(),
                'tokenizer_model': self.ref_model_name
            },
            'conversations': []
        }
        
        successful_generations = 0
        failed_generations = 0
        
        for i in tqdm(range(num_conversations), desc="Generating conversations"):
            conversation = self.generate_conversation(f"conv_{i+1:04d}")
            
            if conversation:
                dataset['conversations'].append(conversation)
                successful_generations += 1
            else:
                failed_generations += 1
                print(f"Failed to generate conversation {i+1}")
        
        # Update dataset info with actual results
        dataset['dataset_info']['successful_generations'] = successful_generations
        dataset['dataset_info']['failed_generations'] = failed_generations
        
        # Save dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {output_file}")
        print(f"Successfully generated: {successful_generations}/{num_conversations} conversations")
        
        # Print statistics
        if successful_generations > 0:
            self._print_dataset_statistics(dataset)
        
        return dataset
    
    def _print_dataset_statistics(self, dataset):
        """Print comprehensive dataset statistics."""
        conversations = dataset['conversations']
        
        # Token statistics
        token_counts = [conv['total_tokens'] for conv in conversations]
        print(f"\nToken Statistics:")
        print(f"  Min tokens: {min(token_counts)}")
        print(f"  Max tokens: {max(token_counts)}")
        print(f"  Mean tokens: {sum(token_counts) / len(token_counts):.1f}")
        
        # QA pair statistics
        qa_counts = [conv['num_qa_pairs'] for conv in conversations]
        print(f"\nQA Pair Statistics:")
        print(f"  Min QA pairs: {min(qa_counts)}")
        print(f"  Max QA pairs: {max(qa_counts)}")
        print(f"  Mean QA pairs: {sum(qa_counts) / len(qa_counts):.1f}")
        
        # Task type distribution
        all_task_types = defaultdict(int)
        for conv in conversations:
            for task_type, count in conv['task_type_distribution'].items():
                all_task_types[task_type] += count
        
        print(f"\nOverall Task Type Distribution:")
        for task_type, count in sorted(all_task_types.items()):
            print(f"  {task_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Generate custom dataset for long memory evaluation')
    parser.add_argument('--num_conversations', type=int, default=10, 
                       help='Number of conversations to generate')
    parser.add_argument('--min_tokens', type=int, default=20000,
                       help='Minimum tokens per conversation')
    parser.add_argument('--max_tokens', type=int, default=25000,
                       help='Maximum tokens per conversation')
    parser.add_argument('--output_file', type=str, default='custom_dataset.json',
                       help='Output file name')
    parser.add_argument('--data_dir', type=str, default='custom_history_data',
                       help='Directory containing the source data')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Model name for tokenization')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = CustomDatasetGenerator(
        data_dir=args.data_dir,
        ref_model_name=args.model_name,
        target_min_tokens=args.min_tokens,
        target_max_tokens=args.max_tokens
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(args.num_conversations, args.output_file)
    
    print("\nDataset generation completed!")


if __name__ == "__main__":
    main() 