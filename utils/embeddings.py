#!/usr/bin/env python3
"""
Shared embeddings module for the INMA RAG system.

This module provides a custom embeddings class that uses Portkey client directly
as specified in the README instructions.
"""

import os
import re
from typing import List
from langchain_core.embeddings import Embeddings
from portkey_ai import Portkey

# Environment variables
GALILEO_API_KEY = os.getenv("GALILEO_API_KEY")
GALILEO_BASE_URL = os.getenv("GALILEO_BASE_URL")


class PortkeyEmbeddings(Embeddings):
    """Custom embeddings class that uses Portkey client directly."""
    
    def __init__(self, model: str = "text-embedding-004", batch_size: int = 100, max_tokens_per_text: int = 8000):
        self.model = model
        self.batch_size = batch_size  # Reduced batch size to handle token limits
        self.max_tokens_per_text = max_tokens_per_text  # Limit tokens per text
        # Initialize Portkey client as specified in README
        self.client = Portkey(
            api_key=GALILEO_API_KEY,
            provider="vertex-ai",
            base_url=GALILEO_BASE_URL,
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (1 token ≈ 4 characters for English/Spanish)."""
        return len(text) // 4
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to stay within token limits."""
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= self.max_tokens_per_text:
            return text
        
        # Truncate to approximately max_tokens_per_text
        max_chars = self.max_tokens_per_text * 4
        truncated = text[:max_chars]
        
        # Try to truncate at a word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # If we can find a space in the last 20%
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Portkey with batching and token limits."""
        try:
            all_embeddings = []
            
            # Process texts in batches to respect vertex-ai limits
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                print(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size} ({len(batch)} texts)")
                
                # Truncate texts to stay within token limits
                truncated_batch = []
                for text in batch:
                    truncated_text = self._truncate_text(text)
                    truncated_batch.append(truncated_text)
                
                # Check total tokens for the batch
                total_chars = sum(len(text) for text in truncated_batch)
                estimated_tokens = total_chars // 4
                
                if estimated_tokens > 18000:  # Leave some buffer
                    print(f"Warning: Batch estimated at {estimated_tokens} tokens, reducing batch size...")
                    # Process in smaller sub-batches
                    sub_batch_size = max(1, self.batch_size // 2)
                    for j in range(0, len(truncated_batch), sub_batch_size):
                        sub_batch = truncated_batch[j:j + sub_batch_size]
                        print(f"  Processing sub-batch {j//sub_batch_size + 1} ({len(sub_batch)} texts)")
                        
                        response = self.client.embeddings.create(
                            input=sub_batch,
                            model=self.model
                        )
                        
                        sub_batch_embeddings = [embedding.embedding for embedding in response.data]
                        all_embeddings.extend(sub_batch_embeddings)
                else:
                    response = self.client.embeddings.create(
                        input=truncated_batch,
                        model=self.model
                    )
                    
                    batch_embeddings = [embedding.embedding for embedding in response.data]
                    all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            print(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Portkey."""
        try:
            # Truncate query text if needed
            truncated_text = self._truncate_text(text)
            
            response = self.client.embeddings.create(
                input=[truncated_text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise
