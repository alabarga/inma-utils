#!/usr/bin/env python3
"""
Create evaluation embeddings for Q&A pairs

This script generates embeddings for question-answer pairs from preguntas.json
and saves them for RAG system evaluation. These embeddings can be used to
calculate semantic similarity between RAG answers and ground truth answers.
"""

import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from utils.embeddings import ProxyEmbeddings


class QAEvaluationEmbeddings:
    """Class to handle Q&A embeddings for evaluation."""
    
    def __init__(self, model: str = None):
        self.embeddings = ProxyEmbeddings(model=model)
        self.qa_pairs = []
        self.question_embeddings = []
        self.answer_embeddings = []
        
    def load_qa_pairs(self, json_file: str):
        """Load Q&A pairs from JSON file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.qa_pairs = json.load(f)
        print(f"Loaded {len(self.qa_pairs)} Q&A pairs")
        
    def generate_embeddings(self):
        """Generate embeddings for all questions and answers."""
        if not self.qa_pairs:
            raise ValueError("No Q&A pairs loaded. Call load_qa_pairs() first.")
        
        print("Generating embeddings for questions...")
        questions = [pair['question'] for pair in self.qa_pairs]
        self.question_embeddings = self.embeddings.embed_documents(questions)
        
        print("Generating embeddings for answers...")
        answers = [pair['answer'] for pair in self.qa_pairs]
        self.answer_embeddings = self.embeddings.embed_documents(answers)
        
        print(f"✅ Generated {len(self.question_embeddings)} question embeddings")
        print(f"✅ Generated {len(self.answer_embeddings)} answer embeddings")
        
    def save_embeddings(self, output_file: str):
        """Save embeddings to pickle file."""
        data = {
            'qa_pairs': self.qa_pairs,
            'question_embeddings': self.question_embeddings,
            'answer_embeddings': self.answer_embeddings,
            'metadata': {
                'total_pairs': len(self.qa_pairs),
                'embedding_dimension': len(self.question_embeddings[0]) if self.question_embeddings else 0,
                'model': self.embeddings.model
            }
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Embeddings saved to: {output_file}")
        
    def load_embeddings(self, input_file: str):
        """Load embeddings from pickle file."""
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        self.qa_pairs = data['qa_pairs']
        self.question_embeddings = data['question_embeddings']
        self.answer_embeddings = data['answer_embeddings']
        
        print(f"✅ Loaded {len(self.qa_pairs)} Q&A pairs with embeddings")
        print(f"📊 Embedding dimension: {data['metadata']['embedding_dimension']}")
        
    def calculate_similarity(self, query_embedding: List[float], answer_embedding: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        query_array = np.array(query_embedding)
        answer_array = np.array(answer_embedding)
        
        # Calculate cosine similarity
        dot_product = np.dot(query_array, answer_array)
        norm_query = np.linalg.norm(query_array)
        norm_answer = np.linalg.norm(answer_array)
        
        if norm_query == 0 or norm_answer == 0:
            return 0.0
            
        return dot_product / (norm_query * norm_answer)
    
    def find_most_similar_question(self, query: str) -> Dict[str, Any]:
        """Find the most similar question to the given query."""
        if not self.question_embeddings:
            raise ValueError("No question embeddings available. Generate embeddings first.")
        
        # Embed the query
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate similarities with all questions
        similarities = []
        for i, question_embedding in enumerate(self.question_embeddings):
            similarity = self.calculate_similarity(query_embedding, question_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True)
        
        best_match_idx = similarities[0][1]
        best_similarity = similarities[0][0]
        
        return {
            'question': self.qa_pairs[best_match_idx]['question'],
            'answer': self.qa_pairs[best_match_idx]['answer'],
            'similarity': best_similarity,
            'index': best_match_idx
        }
    
    def evaluate_rag_answer(self, query: str, rag_answer: str) -> Dict[str, Any]:
        """Evaluate a RAG answer against the ground truth."""
        # Find the most similar question
        best_match = self.find_most_similar_question(query)
        
        # Embed the RAG answer
        rag_embedding = self.embeddings.embed_query(rag_answer)
        
        # Calculate similarity with ground truth answer
        ground_truth_embedding = self.answer_embeddings[best_match['index']]
        answer_similarity = self.calculate_similarity(rag_embedding, ground_truth_embedding)
        
        return {
            'query': query,
            'rag_answer': rag_answer,
            'ground_truth_question': best_match['question'],
            'ground_truth_answer': best_match['answer'],
            'question_similarity': best_match['similarity'],
            'answer_similarity': answer_similarity,
            'overall_score': (best_match['similarity'] + answer_similarity) / 2
        }


def main():
    """Main function to create evaluation embeddings."""
    
    input_file = Path("resources/faqs.json")
    output_file = Path("resources/qa_evaluation_embeddings.pkl")
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found!")
        return
    
    print("🚀 Creating evaluation embeddings for Q&A pairs...")
    
    try:
        # Initialize the evaluation embeddings
        qa_eval = QAEvaluationEmbeddings()
        
        # Load Q&A pairs
        qa_eval.load_qa_pairs(str(input_file))
        
        # Generate embeddings
        qa_eval.generate_embeddings()
        
        # Save embeddings
        qa_eval.save_embeddings(str(output_file))
        
        # Test the evaluation system
        print("\n🧪 Testing evaluation system...")
        test_query = "¿Qué es una vivienda protegida?"
        test_rag_answer = "Una vivienda protegida es aquella que cumple requisitos específicos de superficie, diseño, habitabilidad y precio máximo, recibiendo calificación para acogerse a un régimen de protección pública."
        
        evaluation = qa_eval.evaluate_rag_answer(test_query, test_rag_answer)
        
        print(f"Test Query: {evaluation['query']}")
        print(f"RAG Answer: {evaluation['rag_answer'][:100]}...")
        print(f"Ground Truth Question: {evaluation['ground_truth_question']}")
        print(f"Question Similarity: {evaluation['question_similarity']:.4f}")
        print(f"Answer Similarity: {evaluation['answer_similarity']:.4f}")
        print(f"Overall Score: {evaluation['overall_score']:.4f}")
        
        print(f"\n✅ Evaluation embeddings ready! Use them with the RAG system.")
        
    except Exception as e:
        print(f"❌ Error creating evaluation embeddings: {e}")


if __name__ == "__main__":
    main()
