#!/usr/bin/env python3
"""
Shared embeddings module for the INMA RAG system.
This module provides a proxy embeddings class that can use either Portkey or OpenAI
based on the EMBEDDINGS_PROVIDER environment variable.
"""
import os
from typing import List
from langchain_core.embeddings import Embeddings
from portkey_ai import Portkey
from langchain_openai import OpenAIEmbeddings


class PortkeyEmbeddings(Embeddings):
    """Custom embeddings class that uses Portkey client directly."""
    
    def __init__(self, client: Portkey, model: str = "text-embedding-004", 
                 batch_size: int = 100, max_tokens_per_text: int = 8000):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.max_tokens_per_text = max_tokens_per_text
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens (4 characters per token)."""
        return len(text) // 4
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text to respect token limits."""
        estimated_tokens = self._estimate_tokens(text)
        if estimated_tokens <= self.max_tokens_per_text:
            return text
        
        # Truncate to max_tokens_per_text
        max_chars = self.max_tokens_per_text * 4
        return text[:max_chars]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Portkey with batching and token limits."""
        try:
            # Truncate texts to respect token limits
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(truncated_texts), self.batch_size):
                batch_texts = truncated_texts[i:i + self.batch_size]
                
                # Check total estimated tokens for this batch
                total_tokens = sum(self._estimate_tokens(text) for text in batch_texts)
                
                if total_tokens > 18000:  # Leave some buffer
                    # Split into smaller sub-batches
                    sub_batches = []
                    current_batch = []
                    current_tokens = 0
                    
                    for text in batch_texts:
                        text_tokens = self._estimate_tokens(text)
                        if current_tokens + text_tokens > 18000:
                            if current_batch:
                                sub_batches.append(current_batch)
                            current_batch = [text]
                            current_tokens = text_tokens
                        else:
                            current_batch.append(text)
                            current_tokens += text_tokens
                    
                    if current_batch:
                        sub_batches.append(current_batch)
                    
                    # Process each sub-batch
                    for sub_batch in sub_batches:
                        response = self.client.portkey.embeddings.create(
                            input=sub_batch,
                            model=self.model
                        )
                        all_embeddings.extend([embedding.embedding for embedding in response.data])
                else:
                    # Process normal batch
                    response = self.client.portkey.embeddings.create(
                        input=batch_texts,
                        model=self.model
                    )
                    all_embeddings.extend([embedding.embedding for embedding in response.data])
            
            return all_embeddings
            
        except Exception as e:
            print(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Portkey."""
        try:
            truncated_text = self._truncate_text(text)
            response = self.client.embeddings.create(
                input=[truncated_text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise


class ProxyEmbeddings(Embeddings):
    """Proxy embeddings class that switches between Portkey and OpenAI based on environment variable."""
    
    def __init__(self, model: str = None):
        self.provider = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
        self.model = model or os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002")
        
        if self.provider == "portkey":
            # Initialize Portkey client
            api_key = os.getenv("EMBEDDINGS_API_KEY")
            base_url = os.getenv("EMBEDDINGS_BASE_URL")
            
            if not api_key:
                raise ValueError("EMBEDDINGS_API_KEY environment variable is required for Portkey provider")
            
            client = Portkey(
                api_key=api_key,
                provider="vertex-ai",
                base_url=base_url or "https://eu.aigw.galileo.roche.com/v1",
            )
            
            self.embeddings = PortkeyEmbeddings(client, model=self.model)
            
        elif self.provider == "openai":
            # Initialize OpenAI embeddings
            api_key = os.getenv("EMBEDDINGS_API_KEY")
            base_url = os.getenv("EMBEDDINGS_BASE_URL")
            
            if not api_key:
                raise ValueError("EMBEDDINGS_API_KEY environment variable is required for OpenAI provider")
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                openai_api_base=base_url,
                model=model
            )
            
        else:
            raise ValueError(f"Unsupported embeddings provider: {self.provider}. Use 'portkey' or 'openai'")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using the selected provider."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the selected provider."""
        return self.embeddings.embed_query(text)
