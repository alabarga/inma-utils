#!/usr/bin/env python3
"""
Shared embeddings module for the INMA RAG system.

This module provides a custom embeddings class that uses Portkey client directly
as specified in the README instructions.
"""

import os
from typing import List
from langchain_core.embeddings import Embeddings
from portkey_ai import Portkey

# Environment variables
GALILEO_API_KEY = os.getenv("GALILEO_API_KEY")
GALILEO_BASE_URL = os.getenv("GALILEO_BASE_URL")


class PortkeyEmbeddings(Embeddings):
    """Custom embeddings class that uses Portkey client directly."""
    
    def __init__(self, model: str = "text-embedding-004", batch_size: int = 250):
        self.model = model
        self.batch_size = batch_size  # vertex-ai limit is 250 instances per prediction
        # Initialize Portkey client as specified in README
        self.client = Portkey(
            api_key=GALILEO_API_KEY,
            provider="vertex-ai",
            base_url=GALILEO_BASE_URL,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Portkey with batching."""
        try:
            all_embeddings = []
            
            # Process texts in batches to respect vertex-ai limits
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                print(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size} ({len(batch)} texts)")
                
                response = self.client.embeddings.create(
                    input=batch,
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
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise
