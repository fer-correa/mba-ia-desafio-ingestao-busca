import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

# Carrega variáveis de ambiente
load_dotenv()

def ingest_document():
    # 1. Configurações
    pdf_path = os.getenv("PDF_PATH", "document.pdf")
    connection_string = os.getenv("DATABASE_URL")
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents")

    if not os.path.exists(pdf_path):
        print(f"Erro: Arquivo PDF não encontrado em {pdf_path}")
        return

    print(f"Iniciando ingestão do arquivo: {pdf_path}")

    # 2. Carregamento do PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"PDF carregado: {len(docs)} páginas encontradas.")

    # 3. Chunking (Splitter) conforme SPEC v1.0.1 (1000 caracteres, 150 overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Documento dividido em {len(chunks)} chunks.")

    # 4. Inicialização de Embeddings OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5. Persistência no PGVector
    print(f"Enviando vetores para o banco de dados (coleção: {collection_name})...")
    
    # PGVector da biblioteca langchain-postgres
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection_string,
        use_jsonb=True,
    )

    # Adiciona os documentos ao banco
    vector_store.add_documents(chunks)
    
    print("Sucesso! Ingestão concluída.")

if __name__ == "__main__":
    ingest_document()
