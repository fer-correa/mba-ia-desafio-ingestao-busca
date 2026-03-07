# Ingestão e Busca Semântica com LangChain e Postgres
 - v1.0.1

## Objetivo
Você deve entregar um software capaz de:
 - Ingestão: Ler um arquivo PDF e salvar suas informações em um banco de dados PostgreSQL com extensão pgVector.
 - Busca: Permitir que o usuário faça perguntas via linha de comando (CLI) e receba respostas baseadas apenas no conteúdo do PDF.

## Exemplo no CLI
````

Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.


---

Perguntas fora do contexto:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

````

## Tecnologias obrigatórias

 - Linguagem: Python 3.10+
 - Framework: LangChain (Sintaxe LCEL)
 - Banco de dados: PostgreSQL + pgVector
 - Execução do banco de dados: Docker & Docker Compose

## Pacotes e Modelos

 - Embeddings (OpenAI): `text-embedding-3-small` (via `langchain_openai`)
 - LLM (Gemini): `gemini-1.5-flash` ou `gemini-2.0-flash` (via `langchain_google_genai`)
 - PDF Loader: `PyPDFLoader` (via `langchain_community`)
 - Vector Store: `PGVector` (via `langchain_postgres`)
 - Conexão DB: `postgresql+psycopg://postgres:postgres@localhost:5432/rag`

## Requisitos

1. Ingestão do PDF (`src/ingest.py`)
 - O PDF deve ser dividido em chunks de 1000 caracteres com overlap de 150 usando `RecursiveCharacterTextSplitter`.
 - Cada chunk deve ser convertido em embedding usando OpenAI.
 - Os vetores devem ser armazenados no PostgreSQL com a coleção `pdf_documents`.

2. Consulta via CLI (`src/chat.py` e `src/search.py`)
 - `src/search.py`: Deve conter a lógica de recuperação (retriever) configurada para `k=10`.
 - `src/chat.py`: Script para interação contínua no terminal.
 - Passos ao receber uma pergunta:
   - Buscar os 10 resultados mais relevantes no banco vetorial.
   - Montar o prompt RAG (ver abaixo).
   - Chamar o Gemini via LCEL (`chain = prompt | model | output_parser`).
   - Retornar a resposta ao usuário.

## Prompt a ser utilizado:

````

CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"


````

## Estrutura obrigatória do projeto

````

├── docker-compose.yml
├── requirements.txt      # Dependências (incluindo langchain-openai, langchain-google-genai, langchain-postgres, psycopg)
├── .env.example          # Template das variáveis OPENAI_API_KEY, GOOGLE_API_KEY, DATABASE_URL
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de lógica de busca (retrieval)
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução

````

## Ordem de execução

1. Subir o banco de dados: `docker compose up -d`
2. Configurar `.env` com chaves de API.
3. Executar ingestão: `python src/ingest.py`
4. Rodar o chat: `python src/chat.py`
