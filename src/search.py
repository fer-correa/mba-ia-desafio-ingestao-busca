import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Carrega variáveis de ambiente
load_dotenv()

def get_embeddings():
    """Decide qual provedor de embeddings usar. Requer configuração explícita no .env."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and not openai_key.startswith("your_"):
        model = os.getenv("OPENAI_EMBEDDING_MODEL")
        if not model:
            print("❌ Erro: OPENAI_EMBEDDING_MODEL não configurado no .env")
            sys.exit(1)
        return OpenAIEmbeddings(model=model)
    else:
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL")
        if not model_name:
            print("❌ Erro: GOOGLE_EMBEDDING_MODEL não configurado no .env")
            sys.exit(1)
        clean_model_name = model_name.replace("models/", "")
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{clean_model_name}",
            task_type="retrieval_query"
        )

def get_llm():
    """Decide qual provedor de LLM usar. Requer configuração explícita no .env."""
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if openai_key and not openai_key.startswith("your_"):
        model_name = os.getenv("OPENAI_LLM_MODEL")
        if not model_name:
            print("❌ Erro: OPENAI_LLM_MODEL não configurado no .env")
            sys.exit(1)
        print(f"💡 Usando modelo LLM: OpenAI ({model_name})")
        return ChatOpenAI(model=model_name, temperature=0)
    else:
        model_name = os.getenv("GOOGLE_LLM_MODEL")
        if not model_name:
            print("❌ Erro: GOOGLE_LLM_MODEL não configurado no .env")
            sys.exit(1)
        clean_llm_name = model_name.replace("models/", "")
        print(f"💡 Usando modelo LLM: Google Gemini ({clean_llm_name})")
        return ChatGoogleGenerativeAI(model=clean_llm_name, temperature=0)

def format_docs(docs):
    """Formata os documentos concatenando seu conteúdo."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # 1. Configurações Obrigatórias
    connection_string = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")

    if not connection_string or not collection_name:
        print("❌ Erro: DATABASE_URL ou PG_VECTOR_COLLECTION_NAME não configurados no .env")
        sys.exit(1)
    
    # 2. Inicialização de Embeddings
    embeddings = get_embeddings()
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # 3. Inicialização do Modelo (DINÂMICO)
    model = get_llm()
    
    # 4. Definição do Prompt (conforme SPEC v1.0.2)
    template = """CONTEXTO:
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
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # 5. Construção da Chain via LCEL
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain

if __name__ == "__main__":
    print("Módulo de busca carregado com lógica dinâmica e explícita.")
