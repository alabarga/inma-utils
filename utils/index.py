#!/usr/bin/env python3
"""
FAISS Index Generator for Spanish Laws

This script creates a local FAISS index from the structured JSON law files.
It uses Portkey for embeddings and stores metadata for each chunk.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import uuid

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Portkey configuration
from portkey_ai import Portkey

# Environment variables
GALILEO_API_KEY = os.getenv("GALILEO_API_KEY")
GALILEO_BASE_URL = os.getenv("GALILEO_BASE_URL")


class PortkeyEmbeddings(Embeddings):
    """Custom embeddings class that uses Portkey client directly."""
    
    def __init__(self, client: Portkey, model: str = "text-embedding-004"):
        self.client = client
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Portkey."""
        try:
            response = self.client.portkey.embeddings.create(
                input=texts,
                model=self.model
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Portkey."""
        try:
            response = self.client.portkey.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            raise


class SpanishLawIndexer:
    def __init__(self, resources_dir: str = "resources"):
        self.resources_dir = Path(resources_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize Portkey client as specified in README
        self.client = Portkey(
            api_key=GALILEO_API_KEY,
            provider="vertex-ai",
            base_url=GALILEO_BASE_URL,
        )
        
        # Initialize embeddings using the client
        self.embeddings = PortkeyEmbeddings(self.client, model="text-embedding-004")
        
    def load_law_documents(self) -> List[Dict[str, Any]]:
        """Load all structured JSON law files."""
        documents = []
        
        for json_file in self.resources_dir.glob("*.json"):
            if json_file.name.endswith('.json') and not json_file.name.endswith('_clean.json'):
                print(f"Loading {json_file.name}...")
                with open(json_file, 'r', encoding='utf-8') as f:
                    law_data = json.load(f)
                    documents.append(law_data)
        
        return documents
    
    def extract_chunks_from_law(self, law_data: Dict[str, Any]) -> List[Document]:
        """Extract text chunks from a law document with metadata."""
        chunks = []
        law_id = law_data.get('ley', 'Unknown Law')
        
        # Extract chunks from preámbulo
        for parrafo in law_data.get('preambulo', {}).get('parrafos', []):
            if isinstance(parrafo, dict):
                texto = parrafo.get('texto', '')
                parrafo_id = parrafo.get('id', str(uuid.uuid4()))
            else:
                texto = parrafo
                parrafo_id = str(uuid.uuid4())
            
            if texto.strip():
                # Split the paragraph if it's too long
                sub_chunks = self.text_splitter.split_text(texto)
                for i, chunk in enumerate(sub_chunks):
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            'law_id': law_id,
                            'section': 'preambulo',
                            'parrafo_id': parrafo_id,
                            'chunk_index': i,
                            'total_chunks': len(sub_chunks),
                            'source_file': law_id.replace(' ', '_').replace(',', '').replace('.', '')
                        }
                    ))
        
        # Extract chunks from chapters and articles
        for capitulo in law_data.get('capitulos', []):
            capitulo_num = capitulo.get('capitulo', '')
            capitulo_titulo = capitulo.get('titulo', '')
            
            for articulo in capitulo.get('articulos', []):
                articulo_num = articulo.get('articulo', '')
                articulo_titulo = articulo.get('titulo', '')
                seccion = articulo.get('seccion', '')
                
                for parrafo in articulo.get('parrafos', []):
                    if isinstance(parrafo, dict):
                        texto = parrafo.get('texto', '')
                        parrafo_id = parrafo.get('id', str(uuid.uuid4()))
                    else:
                        texto = parrafo
                        parrafo_id = str(uuid.uuid4())
                    
                    if texto.strip():
                        # Split the paragraph if it's too long
                        sub_chunks = self.text_splitter.split_text(texto)
                        for i, chunk in enumerate(sub_chunks):
                            chunks.append(Document(
                                page_content=chunk,
                                metadata={
                                    'law_id': law_id,
                                    'section': 'articulo',
                                    'capitulo_num': capitulo_num,
                                    'capitulo_titulo': capitulo_titulo,
                                    'articulo_num': articulo_num,
                                    'articulo_titulo': articulo_titulo,
                                    'seccion': seccion,
                                    'parrafo_id': parrafo_id,
                                    'chunk_index': i,
                                    'total_chunks': len(sub_chunks),
                                    'source_file': law_id.replace(' ', '_').replace(',', '').replace('.', '')
                                }
                            ))
        
        # Extract chunks from disposiciones
        disposiciones_types = [
            'disposiciones adicionales',
            'disposiciones transitorias', 
            'disposiciones derogatorias',
            'disposiciones finales'
        ]
        
        for disp_type in disposiciones_types:
            for disposicion in law_data.get(disp_type, []):
                titulo = disposicion.get('titulo', '')
                texto = disposicion.get('texto', '')
                
                if texto.strip():
                    # Split the disposition text if it's too long
                    sub_chunks = self.text_splitter.split_text(texto)
                    for i, chunk in enumerate(sub_chunks):
                        chunks.append(Document(
                            page_content=chunk,
                            metadata={
                                'law_id': law_id,
                                'section': disp_type,
                                'titulo': titulo,
                                'chunk_index': i,
                                'total_chunks': len(sub_chunks),
                                'source_file': law_id.replace(' ', '_').replace(',', '').replace('.', '')
                            }
                        ))
        
        return chunks
    
    def create_faiss_index(self, output_dir: str = "indexes") -> str:
        """Create FAISS index from all law documents."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load all law documents
        law_documents = self.load_law_documents()
        
        # Extract all chunks
        all_chunks = []
        for law_data in law_documents:
            chunks = self.extract_chunks_from_law(law_data)
            all_chunks.extend(chunks)
            print(f"Extracted {len(chunks)} chunks from {law_data.get('ley', 'Unknown Law')}")
        
        print(f"Total chunks: {len(all_chunks)}")
        
        # Create FAISS index using the embeddings
        print("Creating FAISS index...")
        vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
        
        # Save the index
        index_path = output_path / "inma_faiss"
        vectorstore.save_local(str(index_path))
        
        print(f"FAISS index saved to {index_path}")
        
        # Save metadata about the index
        index_metadata = {
            'total_chunks': len(all_chunks),
            'laws_processed': len(law_documents),
            'embedding_model': 'text-embedding-004',
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'index_path': str(index_path)
        }
        
        metadata_path = output_path / "index_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(index_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Index metadata saved to {metadata_path}")
        
        return str(index_path)
    
    def load_faiss_index(self, index_path: str):
        """Load an existing FAISS index."""
        return FAISS.load_local(index_path, self.embeddings)


def main():
    """Main function to create the FAISS index."""
    indexer = SpanishLawIndexer()
    
    try:
        index_path = indexer.create_faiss_index()
        print(f"Successfully created FAISS index at: {index_path}")
        
        # Test the index
        print("\nTesting the index...")
        vectorstore = indexer.load_faiss_index(index_path)
        
        # Test query
        test_query = "¿Cuáles son los requisitos para acceder a una vivienda protegida?"
        results = vectorstore.similarity_search(test_query, k=3)
        
        print(f"\nTest query: {test_query}")
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


if __name__ == "__main__":
    main()
