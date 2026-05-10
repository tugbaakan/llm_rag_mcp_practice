"""
Aşama 3 — Hibrit arama: BM25 (Sparse / anahtar kelime) + yoğun vektör (dense), RRF ile birleştirme.

`rank_bm25.BM25Okapi` ile koleksiyondaki tüm chunk üzerinde BM25 skoru üretilir; Qdrant `query_points`
ile kosinüs yoğun skor sırası alınır. İki sıralama **Reciprocal Rank Fusion (RRF)** ile tek listede
birleştirilir (ayarlama için geleneksel `alpha` karışımı yerine RRF; yaygın hibrit kalıbı).

Önkoşul: Qdrant'ta metin payload (`text` veya `page_content`). Küçük/orta koleksiyonlar için
tüm noktalar `scroll` ile bellekte indekslenir.

Kullanım:
  python stage3_hybrid_rag.py compare "fon risk"
  python stage3_hybrid_rag.py ask "Soru?"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv(Path(__file__).resolve().parent / ".env")

from stage3_naive_rag import (  # noqa: E402
    DEFAULT_QDRANT_URL,
    COLLECTION,
    EMBED_MODEL,
    TOP_K,
    answer_with_groq,
    get_qdrant_client,
)

RRF_K = int(os.environ.get("HYBRID_RRF_K", "60"))
FETCH_K = int(os.environ.get("HYBRID_FETCH_K", "50"))
TOP_OUT = int(os.environ.get("HYBRID_TOP_K", str(TOP_K)))


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _snippet(text: str, max_len: int = 120) -> str:
    t = text.replace("\n", " ").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def tokenize(text: str) -> list[str]:
    """Basit kelime tokenization (Türkçe harfler dahil)."""
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


def _payload_text(payload: dict) -> str:
    return str(payload.get("text") or payload.get("page_content") or "")


def scroll_corpus(
    client,
    collection: str,
    *,
    source_filter: str | None,
) -> list[tuple[Any, str]]:
    """Tüm point id + metin (scroll ile sayfalı)."""
    q_filter = None
    if source_filter:
        q_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    rows: list[tuple[Any, str]] = []
    offset = None
    while True:
        records, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=q_filter,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for r in records:
            pl = dict(r.payload or {})
            t = _payload_text(pl)
            rows.append((r.id, t))
        if next_offset is None:
            break
        offset = next_offset
    return rows


def dense_ranking(
    query: str,
    *,
    client,
    collection: str,
    source_filter: str | None,
    limit: int,
) -> tuple[list[Any], dict[Any, str]]:
    """Yoğun vektör sırasına göre point id listesi (en iyi önce)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    qvec = model.encode([query], convert_to_numpy=True)[0]

    q_filter = None
    if source_filter:
        q_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    res = client.query_points(
        collection_name=collection,
        query=qvec.tolist(),
        limit=limit,
        with_payload=True,
        query_filter=q_filter,
    )
    ordered_ids: list[Any] = []
    id_to_text: dict[Any, str] = {}
    for h in res.points:
        ordered_ids.append(h.id)
        pl = dict(h.payload or {})
        id_to_text[h.id] = _payload_text(pl)
    return ordered_ids, id_to_text


def bm25_ranking(
    query: str,
    corpus_rows: list[tuple[Any, str]],
    top_n: int,
) -> tuple[list[Any], dict[Any, float]]:
    """BM25 skor sırasına göre ilk top_n point id."""
    from rank_bm25 import BM25Okapi

    if not corpus_rows:
        return [], {}

    ids = [r[0] for r in corpus_rows]
    texts = [r[1] for r in corpus_rows]
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    qtok = tokenize(query)
    scores = bm25.get_scores(qtok)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_n]

    ordered_ids = [ids[i] for i in ranked_indices]
    id_to_score = {ids[i]: float(scores[i]) for i in ranked_indices}
    return ordered_ids, id_to_score


def reciprocal_rank_fusion(
    rankings: list[list[Any]],
    k: int = RRF_K,
) -> dict[Any, float]:
    """Birden çok sıralamayı RRF ile birleştirir (id -> birleşik skor)."""
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_top_ids(
    *,
    dense_ids: list[Any],
    bm25_ids: list[Any],
    final_k: int,
) -> list[Any]:
    rrf = reciprocal_rank_fusion([dense_ids, bm25_ids])
    if not rrf:
        return []
    merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in merged[:final_k]]


def cmd_compare(args: argparse.Namespace) -> int:
    client = get_qdrant_client(args.qdrant_url)
    corpus = scroll_corpus(client, args.collection, source_filter=args.source or None)
    if not corpus:
        print("Koleksiyon boş veya bulunamadı.")
        return 1

    id_to_text = {i: t for i, t in corpus}
    n = len(corpus)
    fetch_k = min(args.fetch_k, n)

    dense_ids, _ = dense_ranking(
        args.question,
        client=client,
        collection=args.collection,
        source_filter=args.source or None,
        limit=fetch_k,
    )

    bm25_ids, bm_scores = bm25_ranking(args.question, corpus, top_n=fetch_k)

    hybrid_ids = hybrid_top_ids(
        dense_ids=dense_ids,
        bm25_ids=bm25_ids,
        final_k=args.top_k,
    )

    print(f"Sorgu: «{args.question}»")
    print(f"Koleksiyon: {args.collection} | Chunk sayısı: {n} | RRF sabiti k={RRF_K}")
    print(f"dense/BM25 havuzu: top-{fetch_k} → hibrit çıktı: top-{args.top_k}\n")

    print(f"--- Yoğun (dense) ilk {min(args.top_k, len(dense_ids))} ---")
    for i, pid in enumerate(dense_ids[: args.top_k], 1):
        print(f"  {i}. id={pid}  «{_snippet(id_to_text.get(pid, ''))}»")

    print(f"\n--- BM25 ilk {min(args.top_k, len(bm25_ids))} ---")
    for i, pid in enumerate(bm25_ids[: args.top_k], 1):
        sc = bm_scores.get(pid, 0.0)
        print(f"  {i}. id={pid}  bm25={sc:.4f}  «{_snippet(id_to_text.get(pid, ''))}»")

    print(f"\n--- Hibrit (RRF) top-{args.top_k} ---")
    rrf_scores = reciprocal_rank_fusion([dense_ids, bm25_ids])
    for i, pid in enumerate(hybrid_ids, 1):
        print(f"  {i}. id={pid}  rrf={rrf_scores.get(pid, 0.0):.5f}  «{_snippet(id_to_text.get(pid, ''))}»")

    # Örtüşme: yoğun top-k ile hibrit top-k
    dset = set(dense_ids[: args.top_k])
    hset = set(hybrid_ids)
    inter = len(dset & hset)
    print(f"\nÖrtüşme (dense top-{args.top_k} ∩ hibrit top-{args.top_k}): {inter}/{args.top_k}")

    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    client = get_qdrant_client(args.qdrant_url)
    corpus = scroll_corpus(client, args.collection, source_filter=args.source or None)
    if not corpus:
        print("Koleksiyon boş.")
        return 1

    id_to_text = {i: t for i, t in corpus}
    n = len(corpus)
    fetch_k = min(args.fetch_k, n)

    dense_ids, _ = dense_ranking(
        args.question,
        client=client,
        collection=args.collection,
        source_filter=args.source or None,
        limit=fetch_k,
    )
    bm25_ids, _ = bm25_ranking(args.question, corpus, top_n=fetch_k)

    hybrid_ids = hybrid_top_ids(
        dense_ids=dense_ids,
        bm25_ids=bm25_ids,
        final_k=args.top_k,
    )

    contexts = [id_to_text[pid] for pid in hybrid_ids if pid in id_to_text]
    if not contexts:
        print("Hibrit sonuç boş.")
        return 1

    print("--- Hibrit bağlam (RRF sırası) ---")
    for i, pid in enumerate(hybrid_ids, 1):
        print(f"  [{i}] id={pid}  «{_snippet(id_to_text.get(pid, ''), 100)}»")
    print()

    answer = answer_with_groq(args.question, contexts)
    print("--- LLM cevabı ---")
    print(answer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BM25 + dense hibrit RAG (RRF)")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--source", default="", help="Payload source filtresi")

    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compare", help="Dense vs BM25 vs RRF sıralarını yazdır")
    pc.add_argument("question")
    pc.add_argument("--fetch-k", type=int, default=FETCH_K)
    pc.add_argument("--top-k", type=int, default=TOP_OUT)
    pc.set_defaults(func=cmd_compare)

    pa = sub.add_parser("ask", help="Hibrit retrieval + Groq")
    pa.add_argument("question")
    pa.add_argument("--fetch-k", type=int, default=FETCH_K)
    pa.add_argument("--top-k", type=int, default=TOP_OUT)
    pa.set_defaults(func=cmd_ask)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
