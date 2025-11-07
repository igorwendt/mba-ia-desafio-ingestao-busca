# Ingestão e Busca Semântica com LangChain + PostgreSQL (pgVector)

Sistema de ingestão e busca semântica sobre PDFs utilizando LangChain, PostgreSQL (pgVector) e múltiplos provedores de embeddings (HuggingFace, OpenAI, Google). Suporte a escolha dinâmica de provedor via variável de ambiente ou argumento de linha de comando.

## 🚀 Funcionalidades

- **Ingestão**: Lê PDF e armazena embeddings no PostgreSQL (pgVector)
- **Busca Semântica**: Perguntas respondidas somente com base no conteúdo ingerido
- **Múltiplos Provedores de Embeddings**: `huggingface` (padrão ingest), `openai`, `google`
- **Coleções Isoladas por Provedor**: Evita conflito de dimensões (ex: 384 vs 1536 vs 3072)
- **Contexto Restrito**: Nunca inventa conteúdo fora do PDF

## 📋 Pré-requisitos

- Python 3.9+
- Docker e Docker Compose
- HuggingFace embeddings (gratuitos, default na ingestão)
- (Opcional) Chave OpenAI se usar `--provider openai`
- (Opcional) Chave Google Generative AI se usar `--provider google`

## 🛠️ Instalação

### 1. Clone o repositório
```bash
git clone <seu-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Crie e ative um ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente
Crie um arquivo `.env` (exemplo mínimo):
```bash
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vectordb
PDF_PATH=document.pdf

# Provedor padrão para busca (openai | google) — ingest usa huggingface por padrão
EMBEDDING_PROVIDER=openai

# OpenAI (se usar)
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Google (se usar)
GOOGLE_API_KEY=AIza...
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001

# HuggingFace
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Modelo LLM de resposta (chat)
GPT_MODEL=gpt-5-nano
```

## 🏃‍♂️ Execução

### 1. Subir o banco de dados
```bash
docker compose up -d
```

### 2. Executar ingestão do PDF (gera vetores)
```bash
source venv/bin/activate  # Ativar ambiente virtual
python src/ingest.py                   # Usa huggingface (padrão)
python src/ingest.py --provider openai # Opcional
python src/ingest.py --provider google # Opcional
```

### 3. Rodar o chat / busca
```bash
source venv/bin/activate  # Ativar ambiente virtual
python src/search.py --provider openai -q "Qual o faturamento da delta petroleo epp?"
python src/chat.py                        # usa EMBEDDING_PROVIDER do .env
python src/chat.py --provider google      # força provedor
```

## 💬 Como usar o Chat

Após executar `python src/chat.py`, você pode:

- Fazer perguntas sobre o conteúdo do PDF
- Digitar `ajuda` para ver instruções
- Digitar `sair` para encerrar

### Exemplo de uso:
```
=== Chat com Documentos ===
Digite 'sair' para encerrar o chat
Digite 'ajuda' para ver instruções
----------------------------------------
Chat iniciado com sucesso!
----------------------------------------

Você: Qual o faturamento da Empresa SuperTechIABrazil?
Assistente: O faturamento foi de 10 milhões de reais.

Você: Quantos clientes temos em 2024?
Assistente: Não tenho informações necessárias para responder sua pergunta.
```

## 🏗️ Estrutura do Projeto

```
├── docker-compose.yml      # Configuração do PostgreSQL
├── requirements.txt        # Dependências Python
├── .env.example           # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca semântica
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md            # Este arquivo
```

## 🔧 Configuração

### Variáveis de Ambiente Principais
| Variável | Descrição |
|----------|-----------|
| DATABASE_URL | String de conexão Postgres |
| PDF_PATH | Caminho do PDF a ser ingerido |
| EMBEDDING_PROVIDER | Provedor usado na busca (`openai` ou `google`) |
| OPENAI_API_KEY | API key OpenAI (se usar) |
| OPENAI_EMBEDDING_MODEL | Modelo embeddings OpenAI (default text-embedding-3-small) |
| GOOGLE_API_KEY | API key Google Generative AI (se usar) |
| GOOGLE_EMBEDDING_MODEL | Modelo embeddings Google (default text-embedding-004) |
| HUGGINGFACE_EMBEDDING_MODEL | Modelo HuggingFace (default MiniLM) |
| GPT_MODEL | Modelo LLM para resposta (ex: gpt-5-nano) |

Observação: a ingestão padrão usa HuggingFace para evitar custo. A busca pode ser feita com outro provedor desde que você tenha ingerido previamente para aquele provedor.

### Parâmetros Técnicos
- **Chunk Size**: 1000 caracteres
- **Overlap**: 150 caracteres
- **Top-K** (retrieval): 10 (`similarity_search_with_score`)
- **LLM**: GPT (default `gpt-5-nano` via OpenAI API wrapper)
- **Embeddings suportados**:
	- HuggingFace: sentence-transformers/all-MiniLM-L6-v2 (384 dims)
	- OpenAI: text-embedding-3-small (1536 dims)
	- Google: gemini-embedding-001 (normalmente 768 ou 3072 dims conforme rota)
	- Coleções separadas impedem conflito de dimensão.

### Coleções por Provedor
Ao ingerir criamos coleções distintas no Postgres:
```
documents_huggingface
documents_openai
documents_google
```
Isso evita o erro: `different vector dimensions 384 and 3072`.

### Migração de Versões Anteriores
Se você tinha apenas a coleção `documents` (antiga):
1. Ela ainda existe – não é usada pelo novo código.
2. Re-ingira para cada provedor que deseja usar.
3. (Opcional) Limpeza manual:
```sql
DELETE FROM langchain_pg_collection WHERE name = 'documents';
DELETE FROM langchain_pg_embedding WHERE collection_id NOT IN (SELECT uuid FROM langchain_pg_collection);
```

## 🐛 Solução de Problemas

### Erro de conexão com banco
```bash
# Verificar se o Docker está rodando
docker compose ps

# Recriar o banco se necessário
docker compose down
docker compose up -d
```

### Erro: different vector dimensions 384 and 3072
Você está tentando consultar uma coleção povoada com embeddings de outro provedor. Solução: re-ingira usando o mesmo provedor ou especifique `--provider` correto.

### Erro de API Key
- HuggingFace: não requer chave
- OpenAI: defina `OPENAI_API_KEY`
- Google: defina `GOOGLE_API_KEY`

### Erro de PDF não encontrado
- Verifique se o arquivo `document.pdf` está na raiz do projeto
- Confirme o caminho no arquivo `.env`

## 📚 Tecnologias Utilizadas

- **Python 3.9+**
- **LangChain** - Framework para aplicações com LLM
- **PostgreSQL + pgVector** - Banco vetorial
- **HuggingFace** - Embeddings gratuitos e locais
- **OpenAI** - Embeddings + LLM
- **Google Generative AI** - Embeddings alternativos
- **Docker** - Containerização do banco
- **PyPDF** - Processamento de PDF

## 📄 Licença

Este projeto foi desenvolvido para o desafio MBA Engenharia de Software com IA - Full Cycle.