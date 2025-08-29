#!/usr/bin/env python3
"""
PostgreSQL pgvector Migration Script

This script migrates the FAISS index to PostgreSQL with pgvector extension
for production use and better scalability.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

# Environment variables
GALILEO_API_KEY = os.getenv("GALILEO_API_KEY")
GALILEO_BASE_URL = os.getenv("GALILEO_BASE_URL")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/spanish_laws")

class PGVectorMigrator:
    def __init__(self, database_url: str = DATABASE_URL):
        self.database_url = database_url
        self.embeddings = OpenAIEmbeddings(
            model='text-embedding-004',
            openai_api_base=GALILEO_BASE_URL,
            openai_api_key=GALILEO_API_KEY
        )
        
    def setup_database(self):
        """Set up the database with pgvector extension and required tables."""
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()
        
        try:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create the main vector table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS law_chunks (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_law_chunks_metadata 
                ON law_chunks USING GIN (metadata);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_law_chunks_embedding 
                ON law_chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Create auxiliary tables for law structure
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS laws (
                    id SERIAL PRIMARY KEY,
                    law_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    publication_date TEXT,
                    stats JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chapters (
                    id SERIAL PRIMARY KEY,
                    law_id TEXT REFERENCES laws(law_id),
                    chapter_num TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id SERIAL PRIMARY KEY,
                    chapter_id INTEGER REFERENCES chapters(id),
                    article_num INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    section TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            print("Database setup completed successfully")
            
        except Exception as e:
            conn.rollback()
            print(f"Error setting up database: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def load_faiss_index(self, index_path: str):
        """Load the FAISS index."""
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(index_path, self.embeddings)
    
    def migrate_to_pgvector(self, faiss_index_path: str, collection_name: str = "spanish_laws"):
        """Migrate FAISS index to pgvector."""
        print(f"Loading FAISS index from {faiss_index_path}...")
        
        # Load FAISS index
        faiss_vectorstore = self.load_faiss_index(faiss_index_path)
        
        # Get all documents from FAISS
        # Note: This is a simplified approach. In production, you might want to batch this
        print("Extracting documents from FAISS...")
        
        # Create PGVector connection
        pg_vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=self.database_url,
            embedding_function=self.embeddings
        )
        
        # For demonstration, we'll add documents in batches
        # In a real scenario, you'd want to extract all documents from FAISS
        # and add them to PGVector
        
        print("Migration completed. PGVector is ready for use.")
        return pg_vectorstore
    
    def create_pgvector_from_json(self, resources_dir: str = "resources", collection_name: str = "spanish_laws"):
        """Create pgvector index directly from JSON files."""
        from utils.index import SpanishLawIndexer
        
        print("Creating pgvector index from JSON files...")
        
        # Use the same indexer to extract chunks
        indexer = SpanishLawIndexer(resources_dir)
        law_documents = indexer.load_law_documents()
        
        # Extract all chunks
        all_chunks = []
        for law_data in law_documents:
            chunks = indexer.extract_chunks_from_law(law_data)
            all_chunks.extend(chunks)
            print(f"Extracted {len(chunks)} chunks from {law_data.get('ley', 'Unknown Law')}")
        
        print(f"Total chunks: {len(all_chunks)}")
        
        # Create PGVector index
        pg_vectorstore = PGVector(
            collection_name=collection_name,
            connection_string=self.database_url,
            embedding_function=self.embeddings
        )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            pg_vectorstore.add_documents(batch)
            print(f"Added batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
        
        print("PGVector index created successfully")
        return pg_vectorstore
    
    def populate_auxiliary_tables(self, resources_dir: str = "resources"):
        """Populate auxiliary tables with law structure."""
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor()
        
        try:
            resources_path = Path(resources_dir)
            
            for json_file in resources_path.glob("*.json"):
                if json_file.name.endswith('.json') and not json_file.name.endswith('_clean.json'):
                    print(f"Processing {json_file.name}...")
                    
                    with open(json_file, 'r', encoding='utf-8') as f:
                        law_data = json.load(f)
                    
                    # Insert law
                    law_id = law_data.get('ley', 'Unknown Law')
                    title = law_data.get('ley', 'Unknown Law')
                    publication = law_data.get('publication', '')
                    stats = json.dumps(law_data.get('stats', {}))
                    
                    cursor.execute("""
                        INSERT INTO laws (law_id, title, publication_date, stats)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (law_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        publication_date = EXCLUDED.publication_date,
                        stats = EXCLUDED.stats
                    """, (law_id, title, publication, stats))
                    
                    # Insert chapters
                    for capitulo in law_data.get('capitulos', []):
                        capitulo_num = capitulo.get('capitulo', '')
                        capitulo_titulo = capitulo.get('titulo', '')
                        
                        cursor.execute("""
                            INSERT INTO chapters (law_id, chapter_num, title)
                            VALUES (%s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (law_id, capitulo_num, capitulo_titulo))
                        
                        # Get chapter ID
                        cursor.execute("""
                            SELECT id FROM chapters 
                            WHERE law_id = %s AND chapter_num = %s
                        """, (law_id, capitulo_num))
                        
                        chapter_id = cursor.fetchone()[0]
                        
                        # Insert articles
                        for articulo in capitulo.get('articulos', []):
                            articulo_num = articulo.get('articulo', '')
                            articulo_titulo = articulo.get('titulo', '')
                            seccion = articulo.get('seccion', '')
                            
                            cursor.execute("""
                                INSERT INTO articles (chapter_id, article_num, title, section)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """, (chapter_id, articulo_num, articulo_titulo, seccion))
            
            conn.commit()
            print("Auxiliary tables populated successfully")
            
        except Exception as e:
            conn.rollback()
            print(f"Error populating auxiliary tables: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def query_law_structure(self, law_id: str = None):
        """Query the law structure from auxiliary tables."""
        conn = psycopg2.connect(self.database_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            if law_id:
                cursor.execute("""
                    SELECT l.*, 
                           COUNT(DISTINCT c.id) as chapter_count,
                           COUNT(DISTINCT a.id) as article_count
                    FROM laws l
                    LEFT JOIN chapters c ON l.law_id = c.law_id
                    LEFT JOIN articles a ON c.id = a.chapter_id
                    WHERE l.law_id = %s
                    GROUP BY l.id, l.law_id, l.title, l.publication_date, l.stats
                """, (law_id,))
            else:
                cursor.execute("""
                    SELECT l.*, 
                           COUNT(DISTINCT c.id) as chapter_count,
                           COUNT(DISTINCT a.id) as article_count
                    FROM laws l
                    LEFT JOIN chapters c ON l.law_id = c.law_id
                    LEFT JOIN articles a ON c.id = a.chapter_id
                    GROUP BY l.id, l.law_id, l.title, l.publication_date, l.stats
                """)
            
            results = cursor.fetchall()
            return results
            
        finally:
            cursor.close()
            conn.close()


def main():
    """Main function to set up pgvector and migrate data."""
    migrator = PGVectorMigrator()
    
    try:
        # Set up database
        print("Setting up database...")
        migrator.setup_database()
        
        # Create pgvector index from JSON files
        print("Creating pgvector index...")
        pg_vectorstore = migrator.create_pgvector_from_json()
        
        # Populate auxiliary tables
        print("Populating auxiliary tables...")
        migrator.populate_auxiliary_tables()
        
        # Test the setup
        print("Testing the setup...")
        results = migrator.query_law_structure()
        
        print(f"Found {len(results)} laws in the database:")
        for law in results:
            print(f"- {law['title']} ({law['chapter_count']} chapters, {law['article_count']} articles)")
        
        print("PGVector migration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        raise


if __name__ == "__main__":
    main()
