import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Carrega variáveis de ambiente
load_dotenv()

def format_docs(docs):
    """Formata os documentos concatenando seu conteúdo."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    # 1. Configurações
    connection_string = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")
    
    # 2. Inicialização de Embeddings e Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )
    
    # Configura retriever para k=10 conforme SPEC
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    
    # 3. Inicialização do Modelo Gemini
    # Usando gemini-1.5-flash como padrão estável para a série Flash
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # 4. Definição do Prompt (conforme SPEC v1.0.1)
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
    # Teste rápido de fumaça (requer chaves e banco)
    print("Módulo de busca carregado.")
