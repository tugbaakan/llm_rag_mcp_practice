"""
Aşama 2 — PostgreSQL + pgvector: tablo, vektör ekleme, `<->` ile L2 mesafeye göre sorgu.

Önkoşul: `docker compose up -d` (postgres servisi, varsayılan bağlantı aşağıda).
Postgres’i tarayıcıdan incelemek için aynı compose’daki Adminer: http://localhost:8888
(Sistem: PostgreSQL, Sunucu: postgres, kullanıcı/şifre/DB: practice / practice / vectordb).

`<->` pgvector'da iki vektör arasında Öklid (L2) mesafesini verir; ORDER BY ile
en yakın komşular sıralanır.

Embedding: all-MiniLM-L6-v2 (384 boyut), diğer stage2 scriptleriyle aynı cümleler.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv(Path(__file__).resolve().parent / ".env")

# docker-compose.yml postgres servisi (host 5433, kullanıcı/şifre/DB: practice / practice / vectordb)
DEFAULT_DATABASE_URL = "postgresql://practice:practice@localhost:5433/vectordb"

SENTENCES = [
    "Kediler genelde bağımsız hayvanlardır.",
    "Ev kedileri sık sık uyumayı sever.",
    "Hisse senetleri borsada işlem görür.",
]

QUERY = "Uyuyan evcil kedi davranışı"
TABLE = "practice_docs"


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL).strip()
    if not url:
        print("DATABASE_URL boş.", file=sys.stderr)
        sys.exit(1)

    try:
        conn = psycopg.connect(url, connect_timeout=5)
    except Exception as exc:
        print(
            "PostgreSQL'e bağlanılamadı. Docker Desktop açıkken:\n"
            f"  cd {Path(__file__).resolve().parent}\n"
            "  docker compose up -d\n"
            "İsteğe bağlı: .env içinde DATABASE_URL (varsayılan docker-compose ile uyumlu).\n"
            f"Hata: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    dim = model.get_embedding_dimension()

    doc_embeddings = model.encode(SENTENCES, convert_to_numpy=True)
    query_vec = model.encode([QUERY], convert_to_numpy=True)[0]

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
        cur.execute(
            f"""
            CREATE TABLE {TABLE} (
                id bigserial PRIMARY KEY,
                content text NOT NULL,
                embedding vector({dim}) NOT NULL
            )
            """
        )

        for text, emb in zip(SENTENCES, doc_embeddings, strict=True):
            cur.execute(
                f"INSERT INTO {TABLE} (content, embedding) VALUES (%s, %s)",
                (text, emb),
            )

        # Açıkça `<->` kullanarak en küçük L2 mesafeye göre sıralama
        cur.execute(
            f"""
            SELECT id, content, embedding <-> %s AS distance
            FROM {TABLE}
            ORDER BY embedding <-> %s
            LIMIT 3
            """,
            (query_vec, query_vec),
        )
        rows = cur.fetchall()

    conn.close()

    print(f"Tablo: {TABLE}  |  vektor boyutu: {dim}  |  model: {model_name}")
    print(f"Sorgu metni: «{QUERY}»")
    print("ORDER BY embedding <-> sorgu_vektoru  (L2 mesafesi, düşük = daha yakın):\n")
    for row_id, content, distance in rows:
        print(f"  distance={distance:.4f}  id={row_id}  «{content}»")


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
