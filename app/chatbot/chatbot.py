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
from rank_bm25 import BM25Okapi
import re

# Environment variables
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY")
EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")


class BM25FAQRetriever:
    """BM25-based retriever for FAQ questions."""
    
    def __init__(self, faqs_file: str = "resources/faqs.json", top_k: int = 3):
        self.faqs_file = faqs_file
        self.top_k = top_k
        self.faqs = []
        self.bm25 = None
        self._load_faqs()
        self._build_bm25_index()
    
    def _load_faqs(self):
        """Load FAQs from JSON file."""
        try:
            faqs_path = project_root / self.faqs_file
            with open(faqs_path, 'r', encoding='utf-8') as f:
                self.faqs = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load FAQs from {self.faqs_file}: {e}")
            self.faqs = []
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for BM25 indexing."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into words and remove common Spanish stop words
        stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'una', 'como', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'estan', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mi', 'antes', 'algunos', 'que', 'unos', 'yo', 'otro', 'otras', 'otra', 'el', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tu', 'te', 'ti', 'tu', 'tus', 'ellas', 'nosotras', 'vosotros', 'vosotras', 'os', 'mio', 'mia', 'mios', 'mias', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estas', 'esta', 'estamos', 'estais', 'estan', 'este', 'estes', 'estemos', 'esteis', 'esten', 'estare', 'estaras', 'estara', 'estaremos', 'estareis', 'estaran', 'estaria', 'estarias', 'estariamos', 'estariais', 'estarian', 'estaba', 'estabas', 'estabamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos', 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuvieramos', 'estuvierais', 'estuvieran', 'estuviese', 'estuvieses', 'estuviesemos', 'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada', 'estados', 'estadas', 'estad', 'he', 'has', 'ha', 'hemos', 'habeis', 'han', 'haya', 'hayas', 'hayamos', 'hayais', 'hayan', 'habre', 'habras', 'habra', 'habremos', 'habreis', 'habran', 'habria', 'habrias', 'habriamos', 'habriais', 'habrian', 'habia', 'habias', 'habiamos', 'habiais', 'habian', 'hube', 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubieramos', 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiesemos', 'hubieseis', 'hubiesen', 'habiendo', 'habido', 'habida', 'habidos', 'habidas'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        return ' '.join(words)
    
    def _build_bm25_index(self):
        """Build BM25 index from FAQ questions."""
        if not self.faqs:
            return
        
        # Preprocess all FAQ questions
        processed_questions = []
        for faq in self.faqs:
            processed = self._preprocess_text(faq['question'])
            processed_questions.append(processed.split())
        
        # Build BM25 index
        self.bm25 = BM25Okapi(processed_questions)
    
    def get_relevant_faqs(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant FAQs using BM25."""
        if not self.bm25 or not self.faqs:
            return []
        
        # Preprocess query
        processed_query = self._preprocess_text(query).split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(processed_query)
        
        # Get top-k FAQs
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        
        relevant_faqs = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include FAQs with positive scores
                faq = self.faqs[idx].copy()
                faq['bm25_score'] = scores[idx]
                relevant_faqs.append(faq)
        
        return relevant_faqs


class InmaChat:
    def __init__(self, use_pgvector: bool = False, database_url: Optional[str] = None):
        self.use_pgvector = use_pgvector
        self.conversation_history = []
        
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
        
        # Initialize BM25 FAQ retriever
        self.faq_retriever = BM25FAQRetriever()
        
        # Create RAG prompt with conversation history
        self.prompt = ChatPromptTemplate.from_template("""
        Eres un asistente experto en alquiler de vivienda en Navarra. Tu trabajo es responder preguntas 
        sobre los alquileres en Navarra basándote en la información proporcionada y mantener el contexto de la conversación.
    
        Historial de la conversación:
        {conversation_history}

        Contexto de las leyes:
        {context}

        Preguntas frecuentes (FAQ) relevantes:
        {faq}

        Pregunta actual del usuario: {question}

        Instrucciones:
        1. Valida que la consulta del usuario pertenece al dominio "alquiler de vivienda en Navarra". Si no menciona ámbito geográfico, asume que es en Navarra
        1. Responde únicamente basándote en la información proporcionada en el contexto (historia, leyes, faq)
        2. Si la información no está disponible en el contexto, indícalo claramente
        3. Si la consulta es de "CONTRATO" (celebración, renovación, condiciones esenciales) comprobar si hay datos mínimos del inquilino, arrendador y vivienda; señalar los que falten y proponer preguntas para obtenerlos.
        3. Proporciona respuestas precisas y bien estructuradas. Evita muletillas del tipo "según el contexto proporcionado…". Ve directo al hecho.
        4. Cita las fuentes específicas (ley, artículo, capítulo) cuando sea posible y resume los puntos clave del contexto aportado
        5. Responde en español
        6. Mantén el contexto de la conversación anterior
        7. Si el usuario hace preguntas de seguimiento, responde considerando el historial
        8. No te extiendas en cautelas: sé concreto y útil.

        Respuesta:
        """)
        
        # Create RAG chain
        self.rag_chain = (
            {
                "context": self.retriever,
                "conversation_history": self._format_conversation_history,
                "faq": self._get_relevant_faqs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _get_relevant_faqs(self, question: str) -> str:
        """Get relevant FAQs for the question using BM25."""
        relevant_faqs = self.faq_retriever.get_relevant_faqs(question)
        
        if not relevant_faqs:
            return "No se encontraron preguntas frecuentes relevantes."
        
        formatted_faqs = []
        for faq in relevant_faqs:
            formatted_faqs.append(f"Pregunta: {faq['question']}")
            formatted_faqs.append(f"Respuesta: {faq['answer']}")
            formatted_faqs.append(f"URL: {faq['url']}")
            formatted_faqs.append("---")
        
        return "\n".join(formatted_faqs)
    
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
            
            # Get relevant FAQs
            relevant_faqs = self.faq_retriever.get_relevant_faqs(question)
            
            # Generate response
            response = self.rag_chain.invoke(question)
            
            # Store in conversation history
            self.conversation_history.append({
                'question': question,
                'response': response,
                'sources': [doc.metadata for doc in docs],
                'relevant_faqs': relevant_faqs
            })
            
            return {
                'response': response,
                'sources': [doc.metadata for doc in docs],
                'documents': [doc.page_content for doc in docs],
                'relevant_faqs': relevant_faqs
            }
            
        except Exception as e:
            error_response = f"Error al procesar la consulta: {str(e)}"
            return {
                'response': error_response,
                'sources': [],
                'documents': [],
                'relevant_faqs': [],
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
        chatbot = InmaChat()
        
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
                
                # Show relevant FAQs if available
                if result.get('relevant_faqs'):
                    print(f"\n❓ Preguntas Frecuentes Relacionadas:")
                    for i, faq in enumerate(result['relevant_faqs'][:2], 1):
                        print(f"  {i}. {faq['question']}")
                
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
