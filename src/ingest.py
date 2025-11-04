import warnings
# Suppress urllib3 NotOpenSSLWarning on macOS (must be before any imports that use urllib3)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# Configuration
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vectordb")

# Provider selection (openai, google, huggingface)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

# Models per provider
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

def create_vector_store(provider: str):
    """Create and configure vector store for given provider using provider-specific collection name."""
    try:
        provider = provider.lower()
        if provider not in {"huggingface", "openai", "google"}:
            print(f"Provedor inválido: {provider}. Usando 'huggingface'.")
            provider = "huggingface"

        if provider == "huggingface":
            embeddings = HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            print(f"Usando HuggingFace Embeddings: {HUGGINGFACE_MODEL}")
        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY não definido.")
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(
                model=OPENAI_EMBEDDING_MODEL,
                openai_api_key=OPENAI_API_KEY
            )
            print(f"Usando OpenAI Embeddings: {OPENAI_EMBEDDING_MODEL}")
        else:  # google
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY não definido.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

def ingest_pdf(provider: str):
    """Ingest PDF document into vector database"""
    try:
        # Check if PDF file exists
        if not os.path.exists(PDF_PATH):
            print(f"PDF file not found at: {PDF_PATH}")
            return False
        
        # Load PDF document
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        if not documents:
            print("No documents loaded from PDF")
            return False
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"Split document into {len(splits)} chunks")
        
        # Create vector store
        vector_store = create_vector_store(provider)
        if not vector_store:
            return False
        
        # Add documents to vector store
        vector_store.add_documents(splits)
        
        print("Successfully ingested PDF into vector database")
        return True
        
    except Exception as e:
        print(f"Error ingesting PDF: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestão de PDF em banco vetorial com provedores configuráveis.")
    parser.add_argument("--provider", choices=["huggingface", "openai", "google"], help="Provedor de embedding (sobrepõe EMBEDDING_PROVIDER)")
    args = parser.parse_args()
    chosen = args.provider if args.provider else EMBEDDING_PROVIDER
    success = ingest_pdf(chosen)
    if success:
        print("PDF ingestion completed successfully!")
    else:
        print("PDF ingestion failed!")