"""
Aşama 2 — Embedding nedir, nasıl üretilir?

Embedding: Metin (veya başka veri) sabit boyutlu bir gerçek sayı vektörüne
dönüştürülür; model, anlamsal yakınlığı uzaklık / benzerlik ile temsil eder.

Nasıl üretilir?
- API (OpenAI): Sunucudaki model (ör. text-embedding-3-small) girdiyi işleyip
  vektör döner; API anahtarı gerekir.
- Lokal (sentence-transformers): Önceden eğitilmiş bir model (ör. all-MiniLM-L6-v2)
  ağırlıkları bilgisayarda çalışır; çevrimdışı ve ücretsiz denenebilir.

Bu script birkaç cümleyi embed eder ve çiftler arasında kosinüs benzerliği yazar.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# Örnek cümleler: anlamca yakın / uzak çiftler görmek için
SENTENCES = [
    "Kediler genelde bağımsız hayvanlardır.",
    "Ev kedileri sık sık uyumayı sever.",
    "Hisse senetleri borsada işlem görür.",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """İki vektör arasında kosinüs benzerliği: 1 = aynı yön, 0 = dik, -1 = zıt."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _snippet(text: str, max_len: int = 60) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def print_pair_scores(embeddings: np.ndarray, label: str) -> None:
    n = len(SENTENCES)
    print(f"\n--- {label} — cümle çiftleri (kosinüs benzerliği) ---")
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  [{i}] vs [{j}]: {sim:.4f}")
            print(f"      «{_snippet(SENTENCES[i])}»")
            print(f"      «{_snippet(SENTENCES[j])}»")


def demo_openai() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "\n(OpenAI embedding atlandı: OPENAI_API_KEY yok; .env ile ekleyebilirsiniz.)",
            file=sys.stderr,
        )
        return

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = "text-embedding-3-small"

    response = client.embeddings.create(model=model, input=SENTENCES)
    embeddings = np.array(
        [d.embedding for d in sorted(response.data, key=lambda x: x.index)],
        dtype=np.float64,
    )

    print(f"\nModel: {model} (OpenAI API), boyut: {embeddings.shape[1]}")
    print_pair_scores(embeddings, "OpenAI text-embedding-3-small")


def demo_minilm() -> None:
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(SENTENCES, convert_to_numpy=True, normalize_embeddings=False)

    print(f"\nModel: {model_name} (lokal), boyut: {embeddings.shape[1]}")
    print_pair_scores(embeddings, "all-MiniLM-L6-v2")


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def main() -> None:
    print("Cümleler:")
    for i, s in enumerate(SENTENCES):
        print(f"  [{i}] {s}")

    demo_minilm()
    demo_openai()


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
