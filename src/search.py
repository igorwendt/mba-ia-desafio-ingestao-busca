import warnings
# Suppress urllib3 NotOpenSSLWarning on macOS (must be before any imports that use urllib3)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import os
import argparse
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectordb")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()  # default provider; can be overridden via CLI
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano")
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

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
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def create_vector_store(provider: str):
    """Create and configure the vector store for the chosen embedding provider.

    Uses a provider-specific collection name (documents_<provider>) to avoid dimension conflicts.
    """
    try:
        provider = provider.lower()
        if provider not in {"openai", "google"}:
            print(f"EMBEDDING_PROVIDER inválido: {provider}. Usando 'openai'.")
            provider = "openai"

        # Initialize embeddings according to provider
        if provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY não definido no ambiente.")
            embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            print(f"Usando OpenAI Embeddings: {OPENAI_EMBEDDING_MODEL}")
        else:  # google
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY não definido no ambiente.")
            embeddings = GoogleGenerativeAIEmbeddings(
                model=GOOGLE_EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY
            )
            print(f"Usando Google Embeddings: {GOOGLE_EMBEDDING_MODEL}")

        collection = f"documents_{provider}"
        vector_store = PGVector(
            embeddings=embeddings,
            connection=DATABASE_URL,
            collection_name=collection
        )
        print(f"Coleção usada: {collection}")

        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def format_docs_with_scores(docs_with_scores):
    """Format documents with scores for context"""
    formatted = []
    for doc, score in docs_with_scores:
        formatted.append(f"[Score: {score:.4f}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def search_prompt(question=None, provider: str = EMBEDDING_PROVIDER):
    """Create a search chain for question answering"""
    try:
        # Create vector store
        vector_store = create_vector_store(provider)
        if not vector_store:
            return None
        
        # Initialize provider-specific LLM
        if provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY não definido para uso do LLM OpenAI.")
            llm = ChatOpenAI(
                model=OPENAI_LLM_MODEL,
                openai_api_key=OPENAI_API_KEY,
                temperature=0
            )
            print(f"Using OpenAI LLM model: {OPENAI_LLM_MODEL}\nEmbedding provider: {provider}")
        else:  # google
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY não definido para uso do LLM Google.")
            llm = ChatGoogleGenerativeAI(
                model=GOOGLE_LLM_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0
            )
            print(f"Using Google LLM model: {GOOGLE_LLM_MODEL}\nEmbedding provider: {provider}")
        
        # Create prompt template
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Create a custom retriever function that uses similarity_search_with_score
        def retrieve_with_scores(query):
            docs_with_scores = vector_store.similarity_search_with_score(query, k=10)
            return format_docs_with_scores(docs_with_scores)
        
        # Create chain
        chain = (
            {"contexto": lambda x: retrieve_with_scores(x), "pergunta": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
        
    except Exception as e:
        print(f"Error creating search chain: {e}")
        return None

def search_documents(question, chain=None):
    """Search documents and get answer"""
    try:
        if not chain:
            chain = search_prompt()
        
        if not chain:
            return "Erro: Não foi possível criar a cadeia de busca."
        
        # Get answer
        answer = chain.invoke(question)
        return answer
        
    except Exception as e:
        return f"Erro ao buscar documentos: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste de busca com provedores de embedding.")
    parser.add_argument("--provider", choices=["openai", "google"], help="Provedor de embedding a utilizar (sobrepõe EMBEDDING_PROVIDER).")
    parser.add_argument("--pergunta", "-q", help="Pergunta de teste para executar", default="Teste de busca")
    args = parser.parse_args()

    chosen_provider = args.provider.lower() if args.provider else EMBEDDING_PROVIDER
    chain = search_prompt(provider=chosen_provider)
    if chain:
        question = args.pergunta
        answer = search_documents(question, chain)
        print(f"Pergunta: {question}")
        print(f"Resposta: {answer}")
    else:
        print("Erro ao criar a cadeia de busca")