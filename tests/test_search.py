import pytest
import os
from unittest.mock import MagicMock
from src.search import format_docs, get_rag_chain

def test_format_docs():
    """Valida se a concatenação de documentos funciona corretamente."""
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
    
    docs = [MockDoc("Conteúdo 1"), MockDoc("Conteúdo 2")]
    formatted = format_docs(docs)
    
    assert formatted == "Conteúdo 1\n\nConteúdo 2"

def test_get_rag_chain_initialization(mocker):
    """Valida se a chain RAG é inicializada sem erros com variáveis mockadas."""
    # Mock de ambiente completo
    mock_env = {
        "DATABASE_URL": "postgresql://mock",
        "PG_VECTOR_COLLECTION_NAME": "pdf_documents",
        "GOOGLE_LLM_MODEL": "gemini-1.5-flash",
        "GOOGLE_EMBEDDING_MODEL": "gemini-embedding-001",
        "OPENAI_API_KEY": "your_key"
    }
    mocker.patch('os.getenv', side_effect=lambda k, d=None: mock_env.get(k, d))
    
    mocker.patch('src.search.OpenAIEmbeddings')
    mocker.patch('src.search.GoogleGenerativeAIEmbeddings')
    mocker.patch('src.search.ChatGoogleGenerativeAI')
    mocker.patch('src.search.PGVector')
    
    chain = get_rag_chain()
    assert hasattr(chain, 'invoke')
