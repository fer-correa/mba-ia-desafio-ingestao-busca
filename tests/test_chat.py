import pytest
import sys
from unittest.mock import MagicMock
from src.chat import run_chat

def test_run_chat_exit(mocker):
    """Testa se o chat encerra corretamente ao digitar 'sair' sem lançar erro."""
    # Mock do input para simular o usuário digitando 'sair'
    mocker.patch('builtins.input', side_effect=['sair'])
    # Mock da chain para evitar chamadas reais
    mocker.patch('src.chat.get_rag_chain')
    
    # Executa o chat (deve encerrar o loop via break e retornar None)
    result = run_chat()
    assert result is None

def test_run_chat_init_failure(mocker):
    """Testa se o chat encerra com SystemExit(1) se a chain falhar na inicialização."""
    # Simula erro na inicialização da chain
    mocker.patch('src.chat.get_rag_chain', side_effect=Exception("Erro de conexão"))
    
    # Deve lançar SystemExit(1)
    with pytest.raises(SystemExit) as excinfo:
        run_chat()
    assert excinfo.value.code == 1

def test_run_chat_interaction(mocker):
    """Testa uma interação completa de pergunta e resposta e depois sai."""
    mocker.patch('builtins.input', side_effect=['Qual o faturamento?', 'sair'])
    mock_chain_get = mocker.patch('src.chat.get_rag_chain')
    mock_chain_instance = MagicMock()
    mock_chain_instance.invoke.return_value = "O faturamento foi X."
    mock_chain_get.return_value = mock_chain_instance
    
    run_chat()
    
    assert mock_chain_instance.invoke.called
    assert mock_chain_instance.invoke.call_args[0][0] == 'Qual o faturamento?'
