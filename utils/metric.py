# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import re
import string
from collections import Counter, defaultdict
from rouge import Rouge

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def replace_num(text):
        word_to_number = {
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9"
        }

        pattern = re.compile(r'\b(' + '|'.join(word_to_number.keys()) + r')\b')
        text = pattern.sub(lambda x: word_to_number[x.group()], text)

        return text

    return replace_num(white_space_fix(remove_articles(remove_punc(lower(s)))))


def rouge_score(prediction, ground_truth, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def f1_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    prediction_tokens = pred.split()
    ground_truth_tokens = ref.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(pred, ref, normalize=True):
    if normalize:
        pred, ref = normalize_answer(pred), normalize_answer(ref)
    return pred == ref


def locomo_score(preds, refs, task_types):
    """
    Evaluate LocoMo predictions based on task types
    """
    scores = []
    scores_by_type = {1: [], 2: [], 3: [], 4: [], 5: []}

    for pred, ref, task_type in zip(preds, refs, task_types):
        if pred.endswith("</s>"):
            pred = pred[:-4]
        
        pred = pred.strip()
        if isinstance(ref, (int, float)):
            ref = str(ref)
        ref = ref.strip()
        
        # 1. Multi-Hop
        # 2. Temporal
        # 3. Open-Domain
        # 4. Single-hop
        # 5. Adversarial

        # Handle different task types according to LocoMo evaluation
        if task_type == 1:  # multi-hop
            score = f1_multi_hop(pred, ref)
        elif task_type in [2, 3, 4]:  # temporal, open-domain, single-hop
            # For category 3, split by semicolon and take first part
            if task_type == 3 and ';' in ref:
                ref = ref.split(';')[0].strip()
            score = f1_score(pred, ref)
        elif task_type == 5:  # adversarial
            # Check for correct handling of unanswerable questions
            if 'no information available' in pred.lower() or 'not mentioned' in pred.lower():
                score = 1.0
            else:
                score = 0.0
        else:
            # Default to F1 score
            score = f1_score(pred, ref)
        
        scores.append(score)
        scores_by_type[task_type].append(score)
    
    # Calculate overall and per-type averages
    overall_score = sum(scores) / len(scores) if scores else 0.0
    
    # Print detailed results
    print(f"\nLocoMo Results:")
    print(f"Overall F1: {overall_score:.3f}")
    
    for task_type in [1, 2, 3, 4, 5]:
        if scores_by_type[task_type]:
            avg_score = sum(scores_by_type[task_type]) / len(scores_by_type[task_type])
            print(f"Type {task_type}: {avg_score:.3f} ({len(scores_by_type[task_type])} questions)")
    
    return scores
