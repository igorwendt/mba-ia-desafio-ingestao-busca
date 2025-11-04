import warnings
# Suppress urllib3 NotOpenSSLWarning on macOS (must be before any imports that use urllib3)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import os
import argparse
from dotenv import load_dotenv
from search import search_prompt, search_documents

load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

def main():
    """Main chat interface"""
    print("=== Chat com Documentos ===")
    print("Digite 'sair' para encerrar o chat")
    print("Digite 'ajuda' para ver instruções")
    print("-" * 40)
    
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Chat com documentos usando múltiplos provedores de embeddings/LLM.")
    parser.add_argument("--provider", choices=["openai", "google"], help="Provedor a usar nesta sessão (sobrepõe EMBEDDING_PROVIDER)")
    args, unknown = parser.parse_known_args()
    chosen_provider = args.provider if args.provider else EMBEDDING_PROVIDER

    print(f"Provedor selecionado: {chosen_provider}")

    # Initialize search chain with chosen provider
    chain = search_prompt(provider=chosen_provider)

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        print("Certifique-se de que:")
        print("1. O banco de dados PostgreSQL está rodando")
        print("2. As variáveis de ambiente estão configuradas (.env)")
        print("3. Os documentos foram ingeridos (execute ingest.py primeiro)")
        return
    
    print("Chat iniciado com sucesso!")
    print("-" * 40)
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\nVocê: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['sair', 'exit', 'quit', 'q']:
                print("Encerrando chat...")
                break
            
            # Check for help command
            if user_input.lower() in ['ajuda', 'help', 'h']:
                print_help()
                continue
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Get answer from search
            print("Processando...")
            answer = search_documents(user_input, chain)
            
            # Display answer
            print(f"\nAssistente: {answer}")
            
        except KeyboardInterrupt:
            print("\n\nEncerrando chat...")
            break
        except Exception as e:
            print(f"\nErro: {e}")
            print("Tente novamente ou digite 'sair' para encerrar.")

def print_help():
    """Print help information"""
    print("\n=== AJUDA ===")
    print("Este chat responde perguntas baseadas nos documentos ingeridos.")
    print("\nComandos especiais:")
    print("- 'sair', 'exit', 'quit', 'q': Encerrar o chat")
    print("- 'ajuda', 'help', 'h': Mostrar esta ajuda")
    print("\nDicas:")
    print("- Faça perguntas específicas sobre o conteúdo dos documentos")
    print("- Mude de provedor usando: python src/chat.py --provider openai|google")
    print("- O assistente só responde com base no conteúdo ingerido")
    print("- Se a informação não estiver nos documentos, será informado")
    print("-" * 40)

if __name__ == "__main__":
    main()