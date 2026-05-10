"""
Aşama 3 — Reranking: yoğun retrieval → (isteğe bağlı geniş havuz) → yeniden sıralama → LLM.

- **Cross-encoder** (lokal): `sentence-transformers.CrossEncoder` — API anahtarı gerekmez.
- **Cohere Rerank** (bulut): `COHERE_API_KEY` — `.env` içinde tanımlıysa kullanılır.

Önce yoğun vektör aramasıyla `fetch_k` aday çekilir; reranker bunları yeniden sıralayıp `top_k`
metin seçilir. `compare` komutu baseline (yalnızca vektör skoru, top_k) ile rerank sonucunu ve
örtüşme özetini yazdırır.

Önkoşul: Qdrant'ta indekslenmiş koleksiyon (`stage3_naive_rag.py ingest` veya uyumlu payload).

Kullanım:
  python stage3_rerank_rag.py compare "Yatırım fonu riskleri nelerdir?"
  python stage3_rerank_rag.py ask "Soru?" --rerank cross-encoder
  python stage3_rerank_rag.py ask "Soru?" --rerank cohere
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

FETCH_K = int(os.environ.get("RERANK_FETCH_K", "20"))
TOP_K_FINAL = int(os.environ.get("RERANK_TOP_K", str(TOP_K)))

CROSS_ENCODER_MODEL = os.environ.get(
    "RERANK_CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
COHERE_RERANK_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-v3.5")


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


def _payload_text(payload: dict) -> str:
    return str(
        payload.get("text")
        or payload.get("page_content")
        or ""
    )


def retrieve_pool(
    query: str,
    *,
    qdrant_url: str,
    collection: str,
    source_filter: str | None,
    limit: int,
) -> list[tuple[float, str, dict, str | int]]:
    """(dense_skor, metin, payload, point_id) — naive (`text`) ve LangChain (`page_content`) payload."""
    from sentence_transformers import SentenceTransformer

    client = get_qdrant_client(qdrant_url)
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
    out: list[tuple[float, str, dict, str | int]] = []
    for h in res.points:
        payload = dict(h.payload or {})
        text = _payload_text(payload)
        out.append((float(h.score or 0.0), text, payload, h.id))
    return out


def rerank_cross_encoder(
    query: str,
    pool: list[tuple[float, str, dict, str | int]],
    top_n: int,
) -> list[tuple[float, str, dict, str | int]]:
    from sentence_transformers import CrossEncoder

    if not pool:
        return []

    ce = CrossEncoder(CROSS_ENCODER_MODEL)
    texts = [p[1] for p in pool]
    pairs = [[query, t] for t in texts]
    scores = ce.predict(pairs)
    ranked = sorted(
        zip(scores, pool),
        key=lambda x: float(x[0]),
        reverse=True,
    )[:top_n]
    return [(float(s), row[1], row[2], row[3]) for s, row in ranked]


def rerank_cohere(
    query: str,
    pool: list[tuple[float, str, dict, str | int]],
    top_n: int,
) -> list[tuple[float, str, dict, str | int]]:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY yok; Cohere rerank atlanıyor.", file=sys.stderr)
        return []

    import cohere

    client = cohere.Client(api_key=api_key)
    docs = [p[1] for p in pool]
    resp = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=docs,
        top_n=min(top_n, len(docs)),
    )
    out: list[tuple[float, str, dict, str | int]] = []
    for r in resp.results:
        idx = r.index
        row = pool[idx]
        out.append((float(r.relevance_score), row[1], row[2], row[3]))
    return out


def overlap_ids(a: list[str | int], b: list[str | int]) -> tuple[int, float]:
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    denom = max(len(sa), 1)
    return inter, inter / denom


def cmd_compare(args: argparse.Namespace) -> int:
    fetch_k = args.fetch_k
    top_k = args.top_k

    pool = retrieve_pool(
        args.question,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        source_filter=args.source or None,
        limit=fetch_k,
    )
    if not pool:
        print("Retrieval boş; koleksiyon ve ingest'i kontrol edin.")
        return 1

    baseline = pool[:top_k]
    ce_ranked = rerank_cross_encoder(args.question, pool, top_k)

    print(f"Sorgu: «{args.question}»")
    print(f"Koleksiyon: {args.collection} | Havuz: dense top-{fetch_k} | Gösterim: top-{top_k}\n")

    print("--- Baseline (yalnızca yoğun vektör skoru, ilk k sonuç) ---")
    for i, (sc, text, _pl, pid) in enumerate(baseline, 1):
        print(f"  {i}. id={pid}  dense={sc:.4f}  «{_snippet(text)}»")

    print("\n--- Cross-encoder rerank (aynı havuz üzerinden) ---")
    for i, (sc, text, _pl, pid) in enumerate(ce_ranked, 1):
        print(f"  {i}. id={pid}  ce={sc:.4f}  «{_snippet(text)}»")

    bid = [p[3] for p in baseline]
    cid = [p[3] for p in ce_ranked]
    n_ovl, frac = overlap_ids(bid, cid)
    print(f"\nÖrtüşme@top-{top_k}: {n_ovl}/{top_k} ortak point id (oran={frac:.2f})")

    if args.with_cohere:
        co_ranked = rerank_cohere(args.question, pool, top_k)
        if co_ranked:
            print("\n--- Cohere rerank ---")
            for i, (sc, text, _pl, pid) in enumerate(co_ranked, 1):
                print(f"  {i}. id={pid}  cohere={sc:.4f}  «{_snippet(text)}»")
            oid = [p[3] for p in co_ranked]
            n2, f2 = overlap_ids(bid, oid)
            print(f"\nBaseline vs Cohere örtüşme@top-{top_k}: {n2}/{top_k} (oran={f2:.2f})")
            n3, f3 = overlap_ids(cid, oid)
            print(f"Cross-encoder vs Cohere örtüşme@top-{top_k}: {n3}/{top_k} (oran={f3:.2f})")

    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pool = retrieve_pool(
        args.question,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        source_filter=args.source or None,
        limit=args.fetch_k,
    )
    if not pool:
        print("Retrieval boş.")
        return 1

    mode = args.rerank.strip().lower()
    if mode == "none":
        ranked = pool[: args.top_k]
        contexts = [t for _, t, _, _ in ranked]
    elif mode == "cross-encoder":
        ranked = rerank_cross_encoder(args.question, pool, args.top_k)
        contexts = [t for _, t, _, _ in ranked]
    elif mode == "cohere":
        ranked = rerank_cohere(args.question, pool, args.top_k)
        if not ranked:
            print("Cohere rerank başarısız; COHERE_API_KEY veya cohere paketini kontrol edin.", file=sys.stderr)
            return 1
        contexts = [t for _, t, _, _ in ranked]
    else:
        print(f"Bilinmeyen --rerank: {mode}", file=sys.stderr)
        return 1

    print("--- Seçilen bağlam özetleri (rerank sonrası sıra) ---")
    for i, (_, text, _, pid) in enumerate(ranked, 1):
        print(f"  [{i}] id={pid}  «{_snippet(text, 100)}»")
    print()

    answer = answer_with_groq(args.question, contexts)
    print("--- LLM cevabı ---")
    print(answer)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reranking ile RAG (cross-encoder / Cohere)")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--collection", default=COLLECTION)
    p.add_argument("--source", default="", help="Payload source filtresi (dosya adı)")

    sub = p.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compare", help="Baseline vs rerank sıralamasını ve örtüşmeyi yazdır")
    pc.add_argument("question")
    pc.add_argument("--fetch-k", type=int, default=FETCH_K)
    pc.add_argument("--top-k", type=int, default=TOP_K_FINAL)
    pc.add_argument(
        "--with-cohere",
        action="store_true",
        help="COHERE_API_KEY varsa Cohere rerank karşılaştırması da yazdır",
    )
    pc.set_defaults(func=cmd_compare)

    pa = sub.add_parser("ask", help="Rerank + Groq ile cevap")
    pa.add_argument("question")
    pa.add_argument(
        "--rerank",
        choices=("none", "cross-encoder", "cohere"),
        default="cross-encoder",
    )
    pa.add_argument("--fetch-k", type=int, default=FETCH_K)
    pa.add_argument("--top-k", type=int, default=TOP_K_FINAL)
    pa.set_defaults(func=cmd_ask)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
