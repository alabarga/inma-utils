#!/usr/bin/env python3
"""
Evaluate RAG system responses using pre-generated embeddings

This script loads the evaluation embeddings and provides functions to
evaluate RAG system responses against ground truth answers.
"""

import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from utils.dedeplicate.create_evaluation_embeddings import QAEvaluationEmbeddings


class RAGEvaluator:
    """Class to evaluate RAG system responses."""
    
    def __init__(self, embeddings_file: str = "resources/qa_evaluation_embeddings.pkl"):
        self.qa_eval = QAEvaluationEmbeddings()
        self.embeddings_file = embeddings_file
        self.load_evaluation_data()
        
    def load_evaluation_data(self):
        """Load the evaluation embeddings."""
        if not os.path.exists(self.embeddings_file):
            raise FileNotFoundError(f"Evaluation embeddings file not found: {self.embeddings_file}")
        
        self.qa_eval.load_embeddings(self.embeddings_file)
        
    def evaluate_single_response(self, query: str, rag_answer: str) -> Dict[str, Any]:
        """Evaluate a single RAG response."""
        return self.qa_eval.evaluate_rag_answer(query, rag_answer)
    
    def evaluate_multiple_responses(self, evaluations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate multiple RAG responses and return aggregate metrics."""
        results = []
        
        for eval_item in evaluations:
            result = self.evaluate_single_response(
                eval_item['query'], 
                eval_item['rag_answer']
            )
            results.append(result)
        
        # Calculate aggregate metrics
        question_similarities = [r['question_similarity'] for r in results]
        answer_similarities = [r['answer_similarity'] for r in results]
        overall_scores = [r['overall_score'] for r in results]
        
        return {
            'individual_results': results,
            'aggregate_metrics': {
                'avg_question_similarity': np.mean(question_similarities),
                'avg_answer_similarity': np.mean(answer_similarities),
                'avg_overall_score': np.mean(overall_scores),
                'min_question_similarity': np.min(question_similarities),
                'max_question_similarity': np.max(question_similarities),
                'min_answer_similarity': np.min(answer_similarities),
                'max_answer_similarity': np.max(answer_similarities),
                'total_evaluations': len(results)
            }
        }
    
    def find_similar_questions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the top-k most similar questions to a query."""
        if not self.qa_eval.question_embeddings:
            raise ValueError("No question embeddings available.")
        
        # Embed the query
        query_embedding = self.qa_eval.embeddings.embed_query(query)
        
        # Calculate similarities with all questions
        similarities = []
        for i, question_embedding in enumerate(self.qa_eval.question_embeddings):
            similarity = self.qa_eval.calculate_similarity(query_embedding, question_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity (descending) and get top-k
        similarities.sort(reverse=True)
        top_results = similarities[:top_k]
        
        return [
            {
                'question': self.qa_eval.qa_pairs[idx]['question'],
                'answer': self.qa_eval.qa_pairs[idx]['answer'],
                'similarity': similarity,
                'index': idx
            }
            for similarity, idx in top_results
        ]
    
    def batch_evaluate_from_file(self, input_file: str) -> Dict[str, Any]:
        """Evaluate RAG responses from a JSON file containing query-answer pairs."""
        with open(input_file, 'r', encoding='utf-8') as f:
            evaluations = json.load(f)
        
        return self.evaluate_multiple_responses(evaluations)
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        metrics = evaluation_results['aggregate_metrics']
        results = evaluation_results['individual_results']
        
        report = f"""
# RAG System Evaluation Report

## Summary Statistics
- **Total Evaluations**: {metrics['total_evaluations']}
- **Average Question Similarity**: {metrics['avg_question_similarity']:.4f}
- **Average Answer Similarity**: {metrics['avg_answer_similarity']:.4f}
- **Average Overall Score**: {metrics['avg_overall_score']:.4f}

## Score Ranges
- **Question Similarity**: {metrics['min_question_similarity']:.4f} - {metrics['max_question_similarity']:.4f}
- **Answer Similarity**: {metrics['min_answer_similarity']:.4f} - {metrics['max_answer_similarity']:.4f}

## Individual Results
"""
        
        for i, result in enumerate(results[:10]):  # Show first 10 results
            report += f"""
### Evaluation {i+1}
- **Query**: {result['query']}
- **RAG Answer**: {result['rag_answer'][:100]}...
- **Ground Truth Question**: {result['ground_truth_question']}
- **Question Similarity**: {result['question_similarity']:.4f}
- **Answer Similarity**: {result['answer_similarity']:.4f}
- **Overall Score**: {result['overall_score']:.4f}
"""
        
        if len(results) > 10:
            report += f"\n... and {len(results) - 10} more evaluations.\n"
        
        return report


def main():
    """Main function to demonstrate RAG evaluation."""
    
    embeddings_file = Path("resources/qa_evaluation_embeddings.pkl")
    
    if not embeddings_file.exists():
        print(f"Error: Evaluation embeddings file not found: {embeddings_file}")
        print("Please run utils/create_evaluation_embeddings.py first.")
        return
    
    print("🚀 Loading RAG evaluator...")
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator(str(embeddings_file))
        
        # Example evaluations
        test_evaluations = [
            {
                'query': '¿Qué es una vivienda protegida?',
                'rag_answer': 'Una vivienda protegida es aquella que cumple requisitos específicos de superficie, diseño, habitabilidad y precio máximo, recibiendo calificación para acogerse a un régimen de protección pública.'
            },
            {
                'query': '¿Cuáles son los requisitos para acceder a una vivienda protegida?',
                'rag_answer': 'Los requisitos incluyen límites de ingresos, empadronamiento en Navarra, y no ser titular de otra vivienda adecuada.'
            }
        ]
        
        print("🧪 Running example evaluations...")
        
        # Evaluate individual responses
        for i, eval_item in enumerate(test_evaluations):
            result = evaluator.evaluate_single_response(eval_item['query'], eval_item['rag_answer'])
            print(f"\nEvaluation {i+1}:")
            print(f"  Query: {result['query']}")
            print(f"  Question Similarity: {result['question_similarity']:.4f}")
            print(f"  Answer Similarity: {result['answer_similarity']:.4f}")
            print(f"  Overall Score: {result['overall_score']:.4f}")
        
        # Evaluate multiple responses
        print("\n📊 Running batch evaluation...")
        batch_results = evaluator.evaluate_multiple_responses(test_evaluations)
        
        # Generate report
        report = evaluator.generate_evaluation_report(batch_results)
        print(report)
        
        # Save report
        report_file = Path("resources/rag_evaluation_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Evaluation report saved to: {report_file}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")


if __name__ == "__main__":
    main()
