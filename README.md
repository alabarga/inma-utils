# INMA RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for Spanish laws using LangChain, LangGraph, and modern vector databases.

## 🏗️ Project Structure

```
inma-utils/
├── resources/                    # Law documents and structured data
│   ├── *.md                     # Original markdown law files
│   ├── *.json                   # Structured JSON law files (generated)
│   └── index_metadata.json      # Index statistics and metadata
├── indexes/                     # Vector indexes
│   ├── spanish_laws_faiss/      # FAISS index files
│   └── index_metadata.json      # Index metadata
├── utils/                       # Core utilities
│   ├── parse.py                 # Law document parser (markdown → JSON)
│   ├── index.py                 # FAISS index generator
│   ├── pgvector.py              # PostgreSQL pgvector migration
│   └── clean.py                 # Document cleaning utilities
├── app/                         # Application interfaces
│   ├── chatbot/                 # Chatbot interface
│   │   └── chatbot.py           # Interactive chatbot
│   └── cli/                     # Command-line interface
│       └── rag_cli.py           # RAG CLI tool
├── leyes/                       # Legacy cleaned documents
└── notebooks/                   # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Activate virtual environment
source ~/code/environments/datascience_env/bin/activate

# Set environment variables
export GALILEO_API_KEY="your_gcp_portkey_api_key_for_region"
export GALILEO_BASE_URL="https://eu.aigw.galileo.roche.com/v1"
```

### 2. Parse Law Documents

```bash
# Parse markdown files to structured JSON
python utils/parse.py
```

This creates structured JSON files in `resources/` with:
- UUID-structured paragraphs
- Comprehensive metadata
- Statistics section

### 3. Create Vector Index

```bash
# Create FAISS index
python utils/index.py
```

This generates:
- FAISS index in `indexes/inma_faiss/`
- Index metadata with statistics

### 4. Test the System

```bash
# Interactive CLI
python app/cli/rag_cli.py --interactive

# Single question
python app/cli/rag_cli.py --question "¿Cuáles son los requisitos para vivienda protegida?"

# Chatbot interface
python app/chatbot/chatbot.py
```

## 🔧 Core Components

### Document Processing Pipeline

1. **Parser (`utils/parse.py`)**
   - Converts markdown law files to structured JSON
   - Extracts chapters, articles, and dispositions
   - Adds UUID identifiers and statistics

2. **Indexer (`utils/index.py`)**
   - Creates FAISS vector index from JSON files
   - Uses Portkey client directly for embeddings (as per README)
   - Stores rich metadata for each chunk

3. **Database Migration (`utils/pgvector.py`)**
   - Migrates FAISS index to PostgreSQL with pgvector
   - Creates auxiliary tables for law structure
   - Enables production-ready scalability

### RAG System Architecture

```
User Query → Retriever → Context → LLM → Response
                ↓
         Vector Store (FAISS/pgvector)
                ↓
         Structured Law Documents
```

### Vector Store Options

#### FAISS (Development)
- **Pros**: Fast, local, no database setup
- **Cons**: Limited scalability, no persistence
- **Use case**: Development and testing

#### PostgreSQL + pgvector (Production)
- **Pros**: Scalable, persistent, ACID compliance
- **Cons**: Requires database setup
- **Use case**: Production deployments

## 🛠️ Usage Examples

### CLI Interface

```bash
# Interactive mode
python app/cli/rag_cli.py -i

# Search documents only
python app/cli/rag_cli.py -q "vivienda protegida" -s

# Use pgvector instead of FAISS
python app/cli/rag_cli.py -i --use-pgvector --database-url "postgresql://..."
```

### Chatbot Interface

```bash
python app/chatbot/chatbot.py
```

Features:
- Conversation history
- Source citations
- Interactive commands (`limpiar`, `salir`)

### Database Migration

```bash
# Set up PostgreSQL with pgvector
python utils/pgvector.py
```

This:
- Creates database schema
- Migrates FAISS index to pgvector
- Populates auxiliary tables

## 📊 Data Structure

### JSON Law Format

```json
{
  "ley": "DECRETO FORAL 25/2011...",
  "publication": "Publicado en el Boletín Oficial...",
  "preambulo": {
    "parrafos": [
      {
        "id": "uuid-string",
        "texto": "Paragraph content..."
      }
    ]
  },
  "capitulos": [
    {
      "capitulo": "I",
      "titulo": "Disposiciones generales",
      "articulos": [
        {
          "articulo": 1,
          "titulo": "Objeto.",
          "parrafos": [...],
          "seccion": "Optional section"
        }
      ]
    }
  ],
  "disposiciones adicionales": [...],
  "disposiciones transitorias": [...],
  "disposiciones derogatorias": [...],
  "disposiciones finales": [...],
  "stats": {
    "total_paragraphs": 172,
    "max_length": 4570,
    "min_length": 76,
    "avg_length": 549.0,
    "total_characters": 94428
  }
}
```

### Vector Store Metadata

Each document chunk includes:
- `law_id`: Law identifier
- `section`: Document section (preambulo, articulo, etc.)
- `capitulo_num`: Chapter number
- `articulo_num`: Article number
- `parrafo_id`: Paragraph UUID
- `source_file`: Source file identifier

## 🔌 Integration with LangChain/LangGraph

### Current Implementation
- **LangChain**: RAG pipeline, document processing, vector stores
- **Portkey**: Model access and embeddings (direct client usage with vertex-ai provider)
- **FAISS/pgvector**: Vector storage and retrieval

### Future Enhancements
- **LangGraph**: Multi-agent workflows for complex legal reasoning
- **Advanced RAG**: Hybrid search, re-ranking, query expansion
- **Agentic RAG**: Dynamic retrieval strategies

## 🧪 Testing

### Sample Queries

```bash
# Basic legal questions
"¿Cuáles son los requisitos para acceder a una vivienda protegida?"
"¿Qué establece el artículo 3 sobre unidades familiares?"
"¿Cuándo se considera inadecuada una vivienda?"

# Complex queries
"¿Cómo funciona el baremo de adjudicación de viviendas?"
"¿Qué disposiciones transitorias aplican?"
"¿Cuáles son las sanciones por incumplimiento?"
```

### Performance Metrics

- **Retrieval Accuracy**: Document relevance scoring
- **Response Quality**: Legal accuracy and completeness
- **Response Time**: End-to-end query processing
- **Scalability**: Index size and query performance

## 🔒 Environment Variables

### Portkey Configuration

The system uses Portkey with the vertex-ai provider for embeddings:

```python
client = Portkey(
    api_key="your_gcp_portkey_api_key_for_region",
    provider="vertex-ai",
    base_url="https://eu.aigw.galileo.roche.com/v1",
)
```

### Environment Variables

```bash
# Required
GALILEO_API_KEY=your_gcp_portkey_api_key_for_region
GALILEO_BASE_URL=https://eu.aigw.galileo.roche.com/v1

# Optional (for pgvector)
DATABASE_URL=postgresql://user:password@localhost:5432/spanish_laws
```

## 📈 Statistics

Current law corpus:
- **3 Spanish laws** processed
- **~800 total paragraphs** across all documents
- **Average paragraph length**: 549-782 characters
- **Total characters**: ~580K across all documents

## 🚧 Development

### Adding New Laws

1. Add markdown file to `resources/`
2. Run `python utils/parse.py` to generate JSON
3. Run `python utils/index.py` to update index
4. Test with CLI or chatbot

### Extending Functionality

- **New vector stores**: Implement additional LangChain vector store integrations
- **Advanced retrieval**: Add hybrid search, re-ranking, or query expansion
- **Multi-modal**: Support for legal diagrams, tables, and images
- **API endpoints**: REST API for integration with web applications

## 📚 Dependencies

### Core Dependencies
- `langchain`: RAG framework
- `langchain-openai`: OpenAI integrations
- `langchain-community`: Vector stores and utilities
- `faiss-cpu`: Vector similarity search
- `psycopg2-binary`: PostgreSQL adapter
- `portkey`: Model access layer

### Development Dependencies
- `jupyter`: Notebook analysis
- `pytest`: Testing framework
- `black`: Code formatting
- `flake8`: Linting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Spanish legal documents and their structured format
- LangChain community for the excellent RAG framework
- Portkey for model access infrastructure
- FAISS and pgvector for vector storage solutions
