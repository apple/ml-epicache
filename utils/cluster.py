# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Global embedding models cache
_embedding_models = {}
_qwen_model = None
_qwen_tokenizer = None


def get_embedding_model(embedding_type):
    """
    Get or load embedding model based on embedding_type.
    Models are cached globally to avoid reloading.
    """
    global _embedding_models, _qwen_model, _qwen_tokenizer
    
    if "sentence" in embedding_type:
        if embedding_type not in _embedding_models:
            _embedding_models[embedding_type] = SentenceTransformer('all-MiniLM-L6-v2')
        return _embedding_models[embedding_type]
    elif "qwen" in embedding_type:
        if _qwen_model is None:
            if "4B" in embedding_type:
                model_name = 'Qwen/Qwen3-Embedding-4B'
            else:
                model_name = 'Qwen/Qwen3-Embedding-0.6B'
            _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            _qwen_model = AutoModel.from_pretrained(model_name,
                                                    torch_dtype=torch.bfloat16,
                                                    trust_remote_code=True,
                                                    device_map='auto',
                                                    attn_implementation="flash_attention_2")
        return _qwen_model, _qwen_tokenizer
    
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")


class ClusterManager:
    """
    Manages clustering operations for conversation data with configurable settings.
    """
    
    def __init__(self, embedding_type="sentence_transformer", n_clusters=5, conv_window=4, 
                 medoid_number=None, kv_cache_budget=None, verbose=False):
        """
        Initialize ClusterManager with configuration.
        
        Args:
            embedding_type: Type of embedding to use ("sentence", "bge", "qwen", "llm")
            n_clusters: Number of clusters to create
            conv_window: Number of utterances per conversation window
            medoid_number: Number of top windows to use per cluster (None for all)
            kv_cache_budget: Maximum number of tokens for KV cache (None for no limit)
            verbose: Whether to print detailed progress information
        """
        self.embedding_type = embedding_type
        self.n_clusters = n_clusters
        self.conv_window = conv_window
        self.medoid_number = medoid_number
        self.kv_cache_budget = kv_cache_budget
        self.verbose = verbose
        
        # Store results
        self.all_conversation_results = []
        self.question_cluster_mappings = {}
        
    def print(self, message):
        """Print message only if verbose is enabled."""
        if self.verbose:
            print(message)
    
    def extract_conversation_windows(self, conversation_data: dict) -> list:
        """
        Extract conversation windows from LocoMo conversation data.
        Each window contains conv_window number of utterances as a single string.
        
        Args:
            conversation_data: Dictionary containing conversation data
        
        Returns:
            List of conversation windows, each containing a single string with conv_window utterances
        """
        conversation_windows = []
        
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
        
        # Collect all utterances across all sessions
        all_utterances = []
        for session_num in session_numbers:
            session_key = f'session_{session_num}'
            
            if session_key in conversation_data and conversation_data[session_key]:
                # Add each dialog turn
                for dialog in conversation_data[session_key]:
                    speaker = dialog.get('speaker', 'Unknown')
                    text = dialog.get('clean_text', dialog.get('text', ''))
                    
                    utterance = {
                        'speaker': speaker,
                        'text': text
                    }
                    all_utterances.append(utterance)
        
        # Create conversation windows
        for i in range(0, len(all_utterances), self.conv_window):
            window_utterances = all_utterances[i:i + self.conv_window]
            
            # Build conversation text for this window as a single string
            conversation_text = ""
            for utterance in window_utterances:
                speaker = utterance['speaker']
                text = utterance['text']
                turn = f'{speaker} said, "{text}"\n'
                conversation_text += turn
            
            conversation_windows.append({
                'window_idx': len(conversation_windows),
                'conversation_text': conversation_text,
                'num_utterances': len(window_utterances)
            })
        
        return conversation_windows
    
    def extract_conversation_windows_longmemeval(self, conversation_data: dict) -> list:
        """
        Extract conversation windows from LongMemEval conversation data.
        Each window contains conv_window number of utterances as a single string.
        For long assistant responses, splits them into sentences to manage length.
        
        Args:
            conversation_data: Dictionary containing LongMemEval conversation data
        
        Returns:
            List of conversation windows, each containing a single string with conv_window utterances
        """
        conversation_windows = []
        
        # Collect all utterances across all sessions in conversation_timeline
        all_utterances = []
        
        if 'conversation_timeline' in conversation_data:
            for session_data in conversation_data['conversation_timeline']:
                if 'session' in session_data and session_data['session']:
                    # Add each dialog turn from the session
                    for dialog in session_data['session']:
                        role = dialog.get('role', 'Unknown')
                        content = dialog.get('content', '')
                        
                        # Map roles to speaker names for consistency
                        speaker = 'User' if role == 'user' else 'Assistant'
                        
                        # For assistant responses, split into sentences if single answer is too long
                        if speaker == 'Assistant' and len(content) > 500:  # Threshold for long responses
                            sentences = self._split_into_sentences(content)
                            for sentence in sentences:
                                if len(sentence.strip()) > 10:  # Filter out very short sentences
                                    utterance = {
                                        'speaker': speaker,
                                        'text': sentence.strip()
                                    }
                                    all_utterances.append(utterance)
                        else:
                            # For user messages or short assistant responses, keep as is
                            utterance = {
                                'speaker': speaker,
                                'text': content
                            }
                            all_utterances.append(utterance)
        
        # Create conversation windows
        for i in range(0, len(all_utterances), self.conv_window):
            window_utterances = all_utterances[i:i + self.conv_window]
            
            # Build conversation text for this window as a single string
            conversation_text = ""
            for utterance in window_utterances:
                speaker = utterance['speaker']
                text = utterance['text']
                turn = f'{speaker} said, "{text}"\n'
                conversation_text += turn
            
            conversation_windows.append({
                'window_idx': len(conversation_windows),
                'conversation_text': conversation_text,
                'num_utterances': len(window_utterances)
            })
        
        return conversation_windows
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences using simple regex pattern.
        
        Args:
            text: Text to split into sentences
        
        Returns:
            List of sentences
        """
        import re
        # Split by period, exclamation, question mark followed by space or newline
        sentences = re.split(r'[.!?]+(?:\s+|$)', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def embed_question(self, question: str, model=None) -> np.ndarray:
        """
        Embed a question using the specified embedding method.
        
        Args:
            question: Question text to embed
            model: LLM model for embedding (if embedding_type is "llm")
        
        Returns:
            Normalized question embedding as numpy array
        """
        if "sentence" in self.embedding_type:
            sentence_model = get_embedding_model(self.embedding_type)
            embedding = sentence_model.encode(question)
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
        elif "llm" in self.embedding_type:
            if model is None:
                raise ValueError("Model must be provided for LLM embedding")
            
            inputs = model.tokenizer(question, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model.model.model.embed_tokens(inputs['input_ids'].to(model.device)).float()
                embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
            
        elif "qwen" in self.embedding_type:
            qwen_model, qwen_tokenizer = get_embedding_model(self.embedding_type)
            
            instruction_text = f"\nQuery:{question}"
            inputs = qwen_tokenizer(instruction_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(qwen_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = qwen_model(**inputs)
                # Use last token pooling instead of mean pooling
                embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize embeddings
                embedding = F.normalize(embedding, p=2, dim=1)
                
                # Convert to numpy and ensure 1D
                embedding = embedding.float().cpu().numpy().squeeze()
        
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}. Use 'sentence', 'llm', 'bge', or 'qwen'")
        
        return embedding
    
    def embed_conversations(self, conversation_windows: list, model=None) -> list:
        """
        Embed conversation windows using specified embedding method.
        
        Args:
            conversation_windows: List of conversation windows
            model: LLM model for embedding (if embedding_type is "llm")
        
        Returns:
            List of conversation windows with embeddings added
        """
        embedded_windows = []
        
        if "sentence" in self.embedding_type:
            sentence_model = get_embedding_model(self.embedding_type)
            
            for window in conversation_windows:
                embedding = sentence_model.encode(window['conversation_text'])
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                window_with_embedding = window.copy()
                window_with_embedding['embedding'] = embedding
                embedded_windows.append(window_with_embedding)
            
        elif "llm" in self.embedding_type:
            if model is None:
                raise ValueError("Model must be provided for LLM embedding")
            
            for window in conversation_windows:
                inputs = model.tokenizer(window['conversation_text'], return_tensors="pt", padding=True, truncation=True)
                
                with torch.no_grad():
                    embeddings = model.model.model.embed_tokens(inputs['input_ids'].to(model.device)).float()
                    embedding = embeddings.mean(dim=1).squeeze().cpu().numpy()
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                
                window_with_embedding = window.copy()
                window_with_embedding['embedding'] = embedding
                embedded_windows.append(window_with_embedding)
            
        elif "qwen" in self.embedding_type:
            # Get Qwen3 embedding model from global cache
            qwen_model, qwen_tokenizer = get_embedding_model(self.embedding_type)
            
            # Define instruction task and query for topic analysis
            task = "Analyze conversation content to identify main topics and themes"
            query = "What are the main topics and themes discussed in this conversation?"
            
            for window in conversation_windows:
                # Format instruction with task and query
                instruction_text = f"Task: {task}\nQuery: {query}\n\nDocument: {window['conversation_text']}"
                
                # Tokenize and get embeddings
                inputs = qwen_tokenizer(instruction_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
                inputs = {k: v.to(qwen_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = qwen_model(**inputs)
                    # Use last token pooling instead of mean pooling
                    embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                    
                    # Normalize embeddings
                    embedding = F.normalize(embedding, p=2, dim=1)
                    
                    # Convert to numpy and ensure 1D
                    embedding = embedding.float().cpu().numpy().squeeze()
                
                window_with_embedding = window.copy()
                window_with_embedding['embedding'] = embedding
                embedded_windows.append(window_with_embedding)
        
        else:
            raise ValueError(f"Unknown embedding_type: {self.embedding_type}. Use 'sentence', 'llm', 'bge', or 'qwen'")
        
        return embedded_windows
    
    def cluster_conversations(self, embedded_windows: list) -> dict:
        """
        Cluster conversation windows using normalized embeddings and cosine similarity.
        
        Args:
            embedded_windows: List of conversation windows with embeddings
        
        Returns:
            Dictionary containing clustering results
        """
        # Extract embeddings
        embeddings = np.array([window['embedding'] for window in embedded_windows])
        
        # All embeddings are already normalized, just ensure correct shape
        embeddings_normalized = embeddings
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_normalized)
        
        # Get cluster centroids
        centroids = kmeans.cluster_centers_
        
        # Organize results by cluster
        cluster_results = {}
        for cluster_id in range(self.n_clusters):
            cluster_windows = [window for i, window in enumerate(embedded_windows) if cluster_labels[i] == cluster_id]
            
            # Calculate cosine similarity to centroid for each window in this cluster
            cluster_embeddings = np.array([window['embedding'] for window in cluster_windows])
            # All embeddings are already normalized
            cluster_embeddings_normalized = cluster_embeddings
            
            similarities = cosine_similarity(cluster_embeddings_normalized, [centroids[cluster_id]])
            similarities = similarities.flatten()
            
            # Add similarity scores to windows
            for i, window in enumerate(cluster_windows):
                window['similarity_to_centroid'] = similarities[i]
            
            # Sort by similarity to centroid (most similar first)
            cluster_windows.sort(key=lambda x: x['similarity_to_centroid'], reverse=True)
            
            cluster_results[cluster_id] = {
                'centroid': centroids[cluster_id],
                'windows': cluster_windows,
                'size': len(cluster_windows),
                'avg_similarity': np.mean(similarities)
            }
        
        return {
            'cluster_results': cluster_results,
            'cluster_labels': cluster_labels,
            'n_clusters': self.n_clusters,
            'total_windows': len(embedded_windows)
        }
    
    def make_cluster_prompt(self, cluster_windows: list) -> str:
        """
        Combine conversation segments from a cluster into a single context string.
        
        Args:
            cluster_windows: List of conversation windows in a cluster
        
        Returns:
            Combined conversation text as a single string
        """
        combined_text = ""
        # First sort by similarity_to_centroid (descending) to get top medoid_number windows
        windows = cluster_windows
        if self.medoid_number is not None and self.medoid_number > 0:
            # Sort by similarity and take top medoid_number
            top_windows = sorted(windows, key=lambda x: x['similarity_to_centroid'], reverse=True)[:self.medoid_number]
        else:
            top_windows = windows
        
        # Then sort by window_idx to maintain chronological order
        sorted_windows = sorted(top_windows, key=lambda x: x['window_idx'])
        
        for window in sorted_windows:
            # Add the conversation text from this window
            combined_text += window['conversation_text']
            # Add a separator between windows for clarity
            combined_text += "\n"
        
        return combined_text
            
    def create_question_cluster_mappings(self, dataset, model):
        """
        Create question-to-cluster mappings for all conversations.
        
        Args:
            dataset: Dataset containing questions
            model: LLM model (for llm embedding type)
            data_type: Type of dataset ("locomo", "realtalk", "longmemeval")
        
        Returns:
            Dictionary mapping conversation index to question mappings
        """
        question_cluster_mappings = {}
        
        for conv_idx, conversation in enumerate(dataset):
            
            # Get clustering results for this conversation (from previous clustering)
            if conv_idx < len(self.all_conversation_results):
                clustering_results = self.all_conversation_results[conv_idx]['clustering_results']
                n_clusters = clustering_results['n_clusters']
                
                # Store mappings for this conversation
                conv_mappings = []
                
                # LocoMo/RealTalk structure: separate arrays
                questions = dataset[conv_idx]['question']
                for question_idx, question in enumerate(questions):
                    # Embed the question using specified embedding type
                    question_embedding = self.embed_question(question, model=model)
                    
                    # Calculate similarity with each cluster centroid
                    similarities = []
                    for cluster_id in range(n_clusters):
                        centroid = clustering_results['cluster_results'][cluster_id]['centroid']
                        # Calculate cosine similarity (both vectors are already normalized)
                        similarity = np.dot(question_embedding, centroid)
                        similarities.append(similarity)
                    
                    # Find the closest cluster
                    closest_cluster = np.argmax(similarities)
                    max_similarity = similarities[closest_cluster]
                    
                    # Store mapping
                    mapping = {
                        'question_idx': question_idx,
                        'answer': dataset[conv_idx]['answers'][question_idx],
                        'question': question,
                        'type': dataset[conv_idx]['task_types'][question_idx],
                        'closest_cluster': closest_cluster,
                        'similarity': max_similarity,
                        'all_similarities': similarities
                    }
                    conv_mappings.append(mapping)

                question_cluster_mappings[conv_idx] = conv_mappings        

        return question_cluster_mappings
    
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool the last token of each sequence for embedding generation.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


