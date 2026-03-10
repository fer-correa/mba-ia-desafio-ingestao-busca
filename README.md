# RAG Semântico com LangChain, Postgres (PGVector) e Google Gemini

Este projeto implementa um sistema de **Retrieval-Augmented Generation (RAG)** completo, permitindo a ingestão de documentos PDF e a realização de buscas semânticas via interface de linha de comando (CLI).

## 🚀 Funcionalidades
- **Ingestão Inteligente:** Processamento de PDFs com chunking otimizado (1000/150).
- **Busca Semântica:** Recuperação de contexto baseada em vetores utilizando `pgvector`.
- **Híbrido de IA:** Embeddings via **OpenAI** (`text-embedding-3-small`) e geração de respostas via **Google Gemini** (`gemini-1.5-flash`).
- **Arquitetura Moderna:** Implementado com **LangChain Expression Language (LCEL)** para máxima modularidade.

## 🛠 Pré-requisitos
Antes de iniciar, certifique-se de ter:
- **Docker** e **Docker Compose** instalados (ou Colima no macOS).
- **Python 3.10** ou superior.
- Chaves de API da **OpenAI** e **Google AI Studio**.

## ⚙️ Configuração do Ambiente

### 1. Clonar o Repositório e Preparar o PDF
Coloque o arquivo que deseja indexar na raiz do projeto com o nome `document.pdf`.

### 2. Configurar Variáveis de Ambiente
Copie o arquivo de exemplo e preencha suas chaves:
```bash
cp .env.example .env
```
Edite o arquivo `.env` inserindo sua `OPENAI_API_KEY` e `GOOGLE_API_KEY`.

### 3. Subir a Infraestrutura (PostgreSQL + pgVector)
Inicie o banco de dados via Docker:
```bash
docker compose up -d
```
*Este comando inicializa o Postgres e habilita automaticamente a extensão `vector`.*

### 4. Configurar Ambiente Virtual Python
Crie e ative o ambiente virtual para isolar as dependências:
```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📖 Como Executar

### Passo 1: Ingestão de Dados
Processe o PDF e armazene os vetores no banco de dados:
```bash
python src/ingest.py
```

### Passo 2: Chat CLI
Inicie a interface de conversação para fazer perguntas sobre o documento:
```bash
python src/chat.py
```

## 📂 Estrutura do Projeto
```text
├── src/
│   ├── ingest.py    # Script de processamento e indexação de PDF
│   ├── search.py    # Motor de busca semântica e lógica RAG (LCEL)
│   ├── chat.py      # Interface CLI para interação com o usuário
├── specs/           # Documentação técnica e planos de implementação
├── docker-compose.yml # Definição do banco de dados vetorial
├── requirements.txt   # Dependências do projeto
└── document.pdf       # Arquivo fonte (PDF) para o RAG
```

## ⚖️ Regras de Resposta
O sistema foi configurado com regras rígidas de contexto:
- Responde apenas com base no conteúdo do PDF fornecido.
- Caso a informação não exista no documento, a resposta padrão será: *"Não tenho informações necessárias para responder sua pergunta."*
- Proibido o uso de conhecimento externo ou invenção de fatos.

## 🧪 Testes Automatizados
O projeto possui uma suíte de testes com cobertura mínima de **80%**. Os testes utilizam **Mocks**, o que significa que podem ser executados sem gastar cota de API e sem banco de dados ativo.

### Como executar os testes:
1. Instale as dependências de teste (pytest, pytest-cov, pytest-mock).
2. Execute o comando na raiz do projeto:
```bash
PYTHONPATH=. python -m pytest --cov=src --cov-report=term-missing tests/
```

## 💡 Suporte Multi-Provedor (OpenAI & Google)
O sistema é inteligente na seleção do provedor com base nas chaves fornecidas no `.env`:
- **OpenAI:** Se `OPENAI_API_KEY` estiver preenchida (e não for o valor padrão `your_...`), o sistema utilizará os modelos de Embedding e LLM (GPT) da OpenAI por padrão.
- **Google Gemini:** Caso a chave da OpenAI esteja ausente ou vazia, o sistema utilizará automaticamente o Google Gemini (Embedding e LLM) como fallback.

Você pode configurar modelos específicos nas variáveis `OPENAI_LLM_MODEL` ou `GOOGLE_LLM_MODEL` no seu arquivo `.env`.

## 🧹 Limpeza e Encerramento

Ao finalizar os testes, você pode remover o ambiente virtual e encerrar os serviços do banco de dados com os seguintes comandos:

### 1. Desativar o Ambiente Virtual
Se o ambiente estiver ativo, desative-o:
```bash
deactivate
```

### 2. Remover o Ambiente Virtual (Opcional)
Para remover completamente a pasta `venv`:
```bash
rm -rf venv
```

### 3. Encerrar o Banco de Dados (Docker)
Para parar os containers do PostgreSQL:
```bash
docker compose down
```
*Se desejar apagar também os dados persistidos no banco, utilize: `docker compose down -v`.*

---
> Projeto desenvolvido como parte do desafio do **MBA em Engenharia de Software com IA**.
[Fernando Correa](https://github.com/fer-correa)
