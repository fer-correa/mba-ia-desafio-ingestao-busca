# Plano de Implementação - RAG Semântico (v1.0.1)

Este documento detalha a estratégia de implementação para o sistema de RAG utilizando LangChain, OpenAI (Embeddings), Google Gemini (LLM) e PostgreSQL (Vector Store).

## 1. Definição de Bibliotecas e Versões
Para evitar conflitos de dependência, utilizaremos as seguintes versões estáveis:

| Biblioteca | Versão Sugerida | Finalidade |
| :--- | :--- | :--- |
| `langchain` | `^0.3.0` | Core do Framework |
| `langchain-openai` | `^0.2.0` | Embeddings da OpenAI |
| `langchain-google-genai` | `^2.0.0` | LLM Gemini |
| `langchain-postgres` | `^0.0.12` | Integração PGVector |
| `langchain-community` | `^0.3.0` | Loaders (PyPDFLoader) |
| `psycopg[binary]` | `^3.2.0` | Driver do PostgreSQL |
| `pypdf` | `^5.0.0` | Parse de arquivos PDF |
| `python-dotenv` | `^1.0.1` | Gestão de variáveis de ambiente |

## 2. Matriz de Rastreabilidade (TODO List)

| ID | Requisito da SPEC | Arquivo | Função/Componente |
| :--- | :--- | :--- | :--- |
| **REQ-01** | Ingestão de PDF | `src/ingest.py` | `PyPDFLoader` |
| **REQ-02** | Chunking (1000/150) | `src/ingest.py` | `RecursiveCharacterTextSplitter` |
| **REQ-03** | Embeddings OpenAI | `src/ingest.py` | `OpenAIEmbeddings(model="text-embedding-3-small")` |
| **REQ-04** | Persistência PGVector | `src/ingest.py` | `PGVector.from_documents(collection_name="pdf_documents")` |
| **REQ-05** | Recuperação Semântica (k=10) | `src/search.py` | `vector_store.as_retriever(search_kwargs={"k": 10})` |
| **REQ-06** | Prompt RAG Obrigatório | `src/search.py` | `ChatPromptTemplate.from_template` |
| **REQ-07** | LLM Gemini (LCEL) | `src/search.py` | `ChatGoogleGenerativeAI | StrOutputParser` |
| **REQ-08** | Interface CLI (Loop) | `src/chat.py` | `while True` com `input()` |

## 3. Lógica do Chat e Tratamento de Contexto

### Fluxo de Execução (LCEL):
```python
# Lógica em src/search.py
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
```

### Tratamento "Fora do Contexto":
- O prompt injetado no Gemini contém regras explícitas de "Negative Constraint".
- Se a busca semântica (k=10) não trouxer dados relevantes, o modelo será instruído pelo prompt a responder: *"Não tenho informações necessárias para responder sua pergunta."*
- Não utilizaremos memória de conversa (stateless) conforme sugerido pela estrutura simplificada de busca.

## 4. Estratégia de Validação

1.  **Infra:** Verificar se `docker-compose` subiu o banco e a extensão `vector`.
2.  **Ingestão:** Validar se a tabela `langchain_pg_embedding` foi populada após o `ingest.py`.
3.  **Busca:** Testar perguntas presentes no PDF e perguntas genéricas (ex: "Qual a cor do cavalo branco?") para validar o filtro de contexto.

---
**Próximo Passo:** Instalar dependências e iniciar a implementação do `src/ingest.py`.
