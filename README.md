# Embedding Debug Tool

A small Python tool for creating and searching text embeddings. It uses Ollama to generate vectors locally and Postgres (pgvector) to store and search them.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally
- Docker (for the Postgres database)

## Setup

1. Pull the embedding model:

   ```bash
   ollama pull nomic-embed-text
   ```

2. Install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux
   pip install ollama "psycopg[binary]" pgvector numpy
   ```

3. Start the database:

   ```bash
   docker compose up -d
   ```

4. Check `config.toml` matches your setup:

   ```toml
   [database]
   host = "localhost"
   port = 5432
   dbname = "mydb"
   user = "user"
   password = "password"

   [embedding]
   model = "nomic-embed-text"
   dim = 768
   ```

## Usage

```bash
python loop.py
```

Type text to store it. Prefix with `?` to search.

```
text> Ollama lets you run language models on your own machine
  Stored id=1 (768 dims)

text> The stock market closed higher today
  Stored id=2 (768 dims)

text> ? run an LLM locally
  0.612  Ollama lets you run language models on your own machine
  0.281  The stock market closed higher today

text> quit
```

| Input | Action |
|---|---|
| `some text` | Embed and store it |
| `? some query` | Embed the query and show the closest matches |
| `quit`, `exit`, `q` | Exit |

Scores are cosine similarity. Higher is closer, and the ranking matters more than the absolute number.

