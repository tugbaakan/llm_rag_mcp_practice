"""
Aşama 2 — Qdrant lokal: Docker ile sunucu, koleksiyon, vektör yükleme, sorgu.

Önkoşul: proje kökünde `docker compose up -d` (Qdrant http://localhost:6333).
Bu script all-MiniLM-L6-v2 (384 boyut) ile örnek cümleleri embed eder, Qdrant'a
yazar ve anlamsal arama (cosine) ile en yakın metinleri döndürür.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(Path(__file__).resolve().parent / ".env")

COLLECTION = "llm_rag_practice_demo"
QDRANT_URL = "http://localhost:6333"

SENTENCES = [
    "Kediler genelde bağımsız hayvanlardır.",
    "Ev kedileri sık sık uyumayı sever.",
    "Hisse senetleri borsada işlem görür.",
]

QUERY = "Uyuyan evcil kedi davranışı"


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=5)
        client.get_collections()
    except Exception as exc:
        print(
            "Qdrant'a bağlanılamadı. Docker Desktop'ı açıp tekrar deneyin; sonra:\n"
            f"  cd {Path(__file__).resolve().parent}\n"
            "  docker compose up -d\n"
            f"Hata: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    try:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    except UnexpectedResponse as exc:
        print(f"Koleksiyon oluşturulamadı: {exc}", file=sys.stderr)
        sys.exit(1)

    doc_embeddings = model.encode(SENTENCES, convert_to_numpy=True)
    points = [
        PointStruct(id=i, vector=doc_embeddings[i].tolist(), payload={"text": SENTENCES[i]})
        for i in range(len(SENTENCES))
    ]
    client.upsert(collection_name=COLLECTION, points=points)

    query_vec = model.encode([QUERY], convert_to_numpy=True)[0]
    res = client.query_points(
        collection_name=COLLECTION,
        query=query_vec.tolist(),
        limit=3,
        with_payload=True,
    )
    hits = res.points

    print(f"Koleksiyon: {COLLECTION} (cosine, boyut={dim})")
    print(f"Sorgu: «{QUERY}»\nEn yakın sonuçlar:")
    for h in hits:
        text = (h.payload or {}).get("text", "") if h.payload else ""
        print(f"  skor={h.score:.4f}  id={h.id}  «{text}»")


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
