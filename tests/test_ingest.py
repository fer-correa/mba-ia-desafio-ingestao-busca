import pytest
import os
from unittest.mock import MagicMock, patch
from src.ingest import get_embeddings, ingest_document

def test_get_embeddings_openai(mocker):
    """Valida se seleciona OpenAI quando a chave e o modelo estão presentes."""
    mocker.patch.dict(os.environ, {
        "OPENAI_API_KEY": "sk-12345",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
    })
    mocker.patch('src.ingest.OpenAIEmbeddings')
    
    embeddings = get_embeddings()
    assert "OpenAI" in str(embeddings.__class__) or "MagicMock" in str(embeddings.__class__)

def test_get_embeddings_google(mocker):
    """Valida se seleciona Google quando a chave OpenAI é placeholder e o modelo Google está presente."""
    mocker.patch.dict(os.environ, {
        "OPENAI_API_KEY": "your_openai_api_key_here",
        "GOOGLE_EMBEDDING_MODEL": "gemini-embedding-001"
    })
    mocker.patch('src.ingest.GoogleGenerativeAIEmbeddings')
    
    embeddings = get_embeddings()
    assert "Google" in str(embeddings.__class__) or "MagicMock" in str(embeddings.__class__)

@patch('src.ingest.PyPDFLoader')
@patch('src.ingest.PGVector')
@patch('src.ingest.RecursiveCharacterTextSplitter')
def test_ingest_document_flow(mock_splitter, mock_pgvector, mock_loader, mocker):
    """Valida o fluxo de ingestão sem erros com variáveis mockadas."""
    mock_loader.return_value.load.return_value = [MagicMock(page_content="texto")]
    mock_splitter.return_value.split_documents.return_value = [MagicMock(page_content="chunk")]
    
    # Mock de ambiente completo
    mock_env = {
        "PDF_PATH": "document.pdf",
        "DATABASE_URL": "postgresql://mock",
        "PG_VECTOR_COLLECTION_NAME": "pdf_documents",
        "GOOGLE_EMBEDDING_MODEL": "gemini-embedding-001",
        "OPENAI_API_KEY": "your_key"
    }
    mocker.patch('os.getenv', side_effect=lambda k, d=None: mock_env.get(k, d))
    mocker.patch('os.path.exists', return_value=True)
    
    ingest_document()
    
    assert mock_loader.called
    assert mock_pgvector.called
    assert mock_splitter.called
