import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector

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
        print(f"💡 Usando provedor: OpenAI (Modelo: {model})")
        return OpenAIEmbeddings(model=model)
    else:
        model_name = os.getenv("GOOGLE_EMBEDDING_MODEL")
        if not model_name:
            print("❌ Erro: GOOGLE_EMBEDDING_MODEL não configurado no .env")
            sys.exit(1)
            
        clean_model_name = model_name.replace("models/", "")
        print(f"💡 Usando provedor: Google Gemini (Modelo: {clean_model_name})")
        
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{clean_model_name}",
            task_type="retrieval_document"
        )

def ingest_document():
    # 1. Configurações Obrigatórias
    pdf_path = os.getenv("PDF_PATH")
    connection_string = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")

    if not pdf_path or not connection_string or not collection_name:
        print("❌ Erro: PDF_PATH, DATABASE_URL ou PG_VECTOR_COLLECTION_NAME não configurados no .env")
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"❌ Erro: Arquivo PDF não encontrado em {pdf_path}")
        return

    print(f"Iniciando ingestão do arquivo: {pdf_path}")

    # 2. Carregamento do PDF
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"PDF carregado: {len(docs)} páginas encontradas.")
    except Exception as e:
        print(f"❌ Erro ao carregar PDF: {e}")
        return

    # 3. Chunking (1000/150)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(chunks)} chunks.")

    # 4. Inicialização de Embeddings
    embeddings = get_embeddings()

    # 5. Persistência no PGVector com BATCHING
    print(f"Enviando vetores para o banco de dados (coleção: {collection_name})...")
    
    try:
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
            use_jsonb=True,
        )

        batch_size = 10
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"Processando lote {i//batch_size + 1} de {(len(chunks)-1)//batch_size + 1}...")
            vector_store.add_documents(batch)
            
            if i + batch_size < len(chunks):
                time.sleep(10)

        print("✅ Sucesso! Ingestão concluída.")
    except Exception as e:
        print(f"❌ Erro na conexão com o banco ou persistência: {e}")

if __name__ == "__main__":
    ingest_document()
