import sys
from search import get_rag_chain

def run_chat():
    print("="*50)
    print("SISTEMA DE RAG SEMÂNTICO - MBA IA")
    print("="*50)
    print("Digite 'sair' ou 'exit' para encerrar.\n")
    
    # Inicializa a chain (uma única vez para performance)
    try:
        chain = get_rag_chain()
    except Exception as e:
        print(f"Erro ao inicializar o motor de busca: {e}")
        sys.exit(1)

    while True:
        try:
            print("-" * 30)
            user_question = input("Faça sua pergunta: ").strip()
            
            if not user_question:
                continue
                
            if user_question.lower() in ["sair", "exit", "quit", "q"]:
                print("Encerrando chat. Até logo!")
                break
            
            print("PERGUNTA:", user_question)
            
            # Execução da Chain
            response = chain.invoke(user_question)
            
            print("RESPOSTA:", response)
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nEncerrando chat...")
            break
        except Exception as e:
            print(f"Ocorreu um erro durante a resposta: {e}")

if __name__ == "__main__":
    run_chat()
