#!/usr/bin/env python3
"""
Spanish Laws Chatbot

A simple chatbot interface for the Spanish laws RAG system.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document

# Environment variables
GALILEO_API_KEY = os.getenv("GALILEO_API_KEY")
GALILEO_BASE_URL = os.getenv("GALILEO_BASE_URL")

class SpanishLawChatbot:
    def __init__(self, use_pgvector: bool = False, database_url: Optional[str] = None):
        self.use_pgvector = use_pgvector
        self.conversation_history = []
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            openai_api_base=GALILEO_BASE_URL,
            openai_api_key=GALILEO_API_KEY,
            temperature=0.1
        )
        
        # Initialize vector store
        if use_pgvector and database_url:
            from langchain_community.vectorstores import PGVector
            from ...utils.embeddings import PortkeyEmbeddings
            
            embeddings = PortkeyEmbeddings(model="text-embedding-004")
            
            self.vectorstore = PGVector(
                collection_name="spanish_laws",
                connection_string=database_url,
                embedding_function=embeddings
            )
        else:
            # Use FAISS
            from langchain_community.vectorstores import FAISS
            from ...utils.embeddings import PortkeyEmbeddings
            
            embeddings = PortkeyEmbeddings(model="text-embedding-004")
            
            index_path = project_root / "indexes" / "inma_faiss"
            if index_path.exists():
                self.vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            else:
                raise FileNotFoundError(f"FAISS index not found at {index_path}. Please run utils/index.py first.")
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Create RAG prompt with conversation history
        self.prompt = ChatPromptTemplate.from_template("""
        Eres un asistente experto en leyes españolas. Tu trabajo es responder preguntas sobre las leyes españolas 
        basándote en la información proporcionada y mantener el contexto de la conversación.

        Historial de la conversación:
        {conversation_history}

        Contexto de las leyes:
        {context}

        Pregunta actual del usuario: {question}

        Instrucciones:
        1. Responde únicamente basándote en la información proporcionada en el contexto
        2. Si la información no está disponible en el contexto, indícalo claramente
        3. Proporciona respuestas precisas y bien estructuradas
        4. Cita las fuentes específicas (ley, artículo, capítulo) cuando sea posible
        5. Responde en español
        6. Mantén el contexto de la conversación anterior
        7. Si el usuario hace preguntas de seguimiento, responde considerando el historial

        Respuesta:
        """)
        
        # Create RAG chain
        self.rag_chain = (
            {
                "context": self.retriever,
                "conversation_history": self._format_conversation_history,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_conversation_history(self, question: str) -> str:
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return "No hay historial de conversación previo."
        
        formatted = []
        for entry in self.conversation_history[-6:]:  # Keep last 6 exchanges
            formatted.append(f"Usuario: {entry['question']}")
            formatted.append(f"Asistente: {entry['response']}")
        
        return "\n".join(formatted)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the chatbot with a question and return response with metadata."""
        try:
            # Get relevant documents
            docs = self.retriever.get_relevant_documents(question)
            
            # Generate response
            response = self.rag_chain.invoke(question)
            
            # Store in conversation history
            self.conversation_history.append({
                'question': question,
                'response': response,
                'sources': [doc.metadata for doc in docs]
            })
            
            return {
                'response': response,
                'sources': [doc.metadata for doc in docs],
                'documents': [doc.page_content for doc in docs]
            }
            
        except Exception as e:
            error_response = f"Error al procesar la consulta: {str(e)}"
            return {
                'response': error_response,
                'sources': [],
                'documents': [],
                'error': str(e)
            }
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history.copy()
    
    def search_documents(self, query: str, k: int = 5) -> List[Document]:
        """Search for relevant documents without generating a response."""
        try:
            docs = self.retriever.get_relevant_documents(query)
            return docs
        except Exception as e:
            print(f"Error al buscar documentos: {str(e)}")
            return []


def run_chatbot():
    """Run the chatbot in a simple console interface."""
    print("🤖 Chatbot de Leyes Españolas")
    print("Escribe 'salir' para terminar, 'limpiar' para limpiar el historial")
    print("=" * 60)
    
    try:
        chatbot = SpanishLawChatbot()
        
        while True:
            try:
                question = input("\n👤 Tú: ").strip()
                
                if question.lower() in ['salir', 'exit', 'quit']:
                    print("¡Hasta luego! 👋")
                    break
                
                if question.lower() in ['limpiar', 'clear']:
                    chatbot.clear_history()
                    print("✅ Historial limpiado")
                    continue
                
                if not question:
                    continue
                
                print("🤖 Asistente: Procesando...")
                
                # Get response
                result = chatbot.query(question)
                
                print(f"\n🤖 Asistente: {result['response']}")
                
                # Show sources if available
                if result['sources']:
                    print(f"\n📚 Fuentes:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        law_id = source.get('law_id', 'Desconocida')
                        section = source.get('section', 'Desconocida')
                        if source.get('articulo_num'):
                            print(f"  {i}. {law_id} - {section} - Artículo {source['articulo_num']}")
                        else:
                            print(f"  {i}. {law_id} - {section}")
                
            except KeyboardInterrupt:
                print("\n¡Hasta luego! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Error inicializando el chatbot: {e}")
        print("Asegúrate de que:")
        print("1. Las variables de entorno GALILEO_API_KEY y GALILEO_BASE_URL estén configuradas")
        print("2. El índice FAISS haya sido creado ejecutando utils/index.py")
        sys.exit(1)


if __name__ == "__main__":
    run_chatbot()
