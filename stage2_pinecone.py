"""
Aşama 2 — Pinecone (cloud): serverless index, 100+ vektör, nearest-neighbor sorgu.

Weaviate yerine Pinecone seçildi; ikisi de vektör DB cloud deneyimi için benzer rolde.

Kurulum: https://www.pinecone.io/ ücretsiz hesap → API key.
Ortam: PINECONE_API_KEY (zorunlu). İsteğe bağlı: PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION.

Index yoksa oluşturulur (384 boyut, cosine); varsa boyut uyumu sizin sorumluluğunuzda.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_INDEX = "llm-rag-practice-demo"
NUM_VECTORS = 120
EMBED_DIM = 384
QUERY_TEXT = "Evdeki kedi gün içinde çoğu zaman uyur mu?"

TOPICS = [
    "Kedilerin uyku düzeni ve ev içi davranışları.",
    "Köpek eğitimi ve parkta egzersiz önerileri.",
    "Borsada volatilite ve hisse seçimi notları.",
    "PostgreSQL indeksleri ve sorgu optimizasyonu.",
    "Makine öğrenmesinde embedding ve vektör arama.",
]


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _wait_index_ready(pc: Pinecone, name: str, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = pc.describe_index(name)
        if info.status.ready:
            return
        time.sleep(3)
    print("Index hazır olma süresi aşıldı.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    api_key = (os.environ.get("PINECONE_API_KEY") or "").strip()
    if not api_key:
        print(
            "PINECONE_API_KEY tanımlı değil. https://app.pinecone.io/ → API Keys; "
            ".env içine PINECONE_API_KEY=... ekleyin (.env.example'a bakın).",
            file=sys.stderr,
        )
        sys.exit(1)

    index_name = (os.environ.get("PINECONE_INDEX") or DEFAULT_INDEX).strip()
    cloud = (os.environ.get("PINECONE_CLOUD") or "aws").strip()
    region = (os.environ.get("PINECONE_REGION") or "us-east-1").strip()

    from sentence_transformers import SentenceTransformer

    texts = [f"[kayıt {i}] {TOPICS[i % len(TOPICS)]}" for i in range(NUM_VECTORS)]
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_embedding_dimension()
    if dim != EMBED_DIM:
        print(f"Beklenen boyut {EMBED_DIM}, model {dim} döndürdü.", file=sys.stderr)
        sys.exit(1)

    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"Index oluşturuldu: {index_name} ({cloud}/{region}). Hazır olması bekleniyor…")
        _wait_index_ready(pc, index_name)
    else:
        print(f"Mevcut index kullanılıyor: {index_name}")

    index = pc.Index(index_name)

    rows = [
        {"id": str(i), "values": embeddings[i].astype(float).tolist(), "metadata": {"idx": i, "text": texts[i][:500]}}
        for i in range(NUM_VECTORS)
    ]
    index.upsert(vectors=rows, batch_size=100, show_progress=False)
    print(f"Upsert tamam: {NUM_VECTORS} vektör.")

    q = model.encode([QUERY_TEXT], convert_to_numpy=True, normalize_embeddings=True)[0]
    # Pinecone API: top_k en az 2 olmalı (SDK doğrulaması)
    res = index.query(vector=q.astype(float).tolist(), top_k=5, include_metadata=True)

    print(f"\nSorgu: «{QUERY_TEXT}»\nEn yakın 5 komşu (cosine skor — yüksek = daha yakın):")
    for m in res.matches:
        meta = m.metadata or {}
        snippet = meta.get("text", "")
        disp = snippet if len(snippet) <= 90 else snippet[:90] + "…"
        print(f"  id={m.id}  score={m.score:.4f}  «{disp}»")


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
