import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_embeddings():
    """Mock para funções de embedding."""
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 1536]
    mock.embed_query.return_value = [0.1] * 1536
    return mock

@pytest.fixture
def mock_vector_store():
    """Mock para o PGVector."""
    mock = MagicMock()
    return mock

@pytest.fixture
def mock_llm():
    """Mock para o modelo Gemini."""
    mock = MagicMock()
    mock.invoke.return_value = "Resposta mockada do contexto."
    return mock
