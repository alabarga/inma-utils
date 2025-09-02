#!/usr/bin/env python3
"""
RAG CLI for Spanish Laws

A simple command-line interface for testing the RAG system with Spanish laws.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Environment variables
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-2025-04-14")

class InmaRAG:
    def __init__(self, use_pgvector: bool = False, database_url: Optional[str] = None):
        self.use_pgvector = use_pgvector
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_base=LLM_BASE_URL,
            openai_api_key=LLM_API_KEY,
            temperature=0.1
        )
        
        # Initialize vector store
        if use_pgvector and database_url:
            from langchain_community.vectorstores import PGVector
            from utils.embeddings import ProxyEmbeddings
            
            embeddings = ProxyEmbeddings(model=EMBEDDINGS_MODEL)
            
            self.vectorstore = PGVector(
                collection_name="spanish_laws",
                connection_string=database_url,
                embedding_function=embeddings
            )
        else:
            # Use FAISS
            from langchain_community.vectorstores import FAISS
            from utils.embeddings import ProxyEmbeddings
            
            embeddings = ProxyEmbeddings(model=EMBEDDINGS_MODEL)
            
            index_path = project_root / "indexes" / "inma_faiss"
            if index_path.exists():
                self.vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            else:
                raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run utils/index.py first.")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create RAG prompt
        self.prompt = ChatPromptTemplate.from_template("""
        Eres un asistente experto en leyes españolas. Tu trabajo es responder preguntas sobre las leyes españolas 
        basándote en la información proporcionada.

        Contexto de las leyes:
        {context}

        Pregunta del usuario: {question}

        Instrucciones:
        1. Responde únicamente basándote en la información proporcionada en el contexto
        2. Si la información no está disponible en el contexto, indícalo claramente
        3. Proporciona respuestas precisas y bien estructuradas
        4. Cita las fuentes específicas (ley, artículo, capítulo) cuando sea posible
        5. Responde en español

        Respuesta:
        """)
        
        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        """Query the RAG system with a question."""
        try:
            response = self.rag_chain.invoke(question)
            return response
        except Exception as e:
            return f"Error al procesar la consulta: {str(e)}"
    
    def search_documents(self, query: str, k: int = 5) -> List:
        """Search for relevant documents without generating a response."""
        try:
            docs = self.retriever.get_relevant_documents(query)
            return docs
        except Exception as e:
            print(f"Error al buscar documentos: {str(e)}")
            return []


def main():
    parser = argparse.ArgumentParser(description="RAG CLI for Spanish Laws")
    parser.add_argument("--question", "-q", type=str, help="Question to ask about Spanish laws")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--search-only", "-s", action="store_true", help="Only search documents, don't generate response")
    parser.add_argument("--use-pgvector", action="store_true", help="Use PostgreSQL with pgvector instead of FAISS")
    parser.add_argument("--database-url", type=str, help="Database URL for pgvector")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        rag = InmaRAG(use_pgvector=args.use_pgvector, database_url=args.database_url)
        
        if args.interactive:
            print("RAG CLI para Leyes Españolas")
            print("Escribe 'salir' para terminar")
            print("-" * 50)
            
            while True:
                try:
                    question = input("\nPregunta: ").strip()
                    if question.lower() in ['salir', 'exit', 'quit']:
                        break
                    
                    if not question:
                        continue
                    
                    if args.search_only:
                        docs = rag.search_documents(question, args.k)
                        print(f"\nEncontrados {len(docs)} documentos relevantes:")
                        for i, doc in enumerate(docs, 1):
                            print(f"\n--- Documento {i} ---")
                            print(f"Contenido: {doc.page_content[:300]}...")
                            print(f"Metadatos: {doc.metadata}")
                    else:
                        response = rag.query(question)
                        print(f"\nRespuesta: {response}")
                        
                except KeyboardInterrupt:
                    print("\n¡Hasta luego!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
        
        elif args.question:
            if args.search_only:
                docs = rag.search_documents(args.question, args.k)
                print(f"Encontrados {len(docs)} documentos relevantes para: '{args.question}'")
                for i, doc in enumerate(docs, 1):
                    print(f"\n--- Documento {i} ---")
                    print(f"Contenido: {doc.page_content[:300]}...")
                    print(f"Metadatos: {doc.metadata}")
            else:
                response = rag.query(args.question)
                print(f"Pregunta: {args.question}")
                print(f"Respuesta: {response}")
        
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"Error inicializando el sistema RAG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
