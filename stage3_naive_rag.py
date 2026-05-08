"""
Aşama 3 — Naive RAG: PDF chunk → embed → Qdrant; sorgu → retrieval → LLM cevabı.

Önkoşullar:
- Qdrant: `docker compose up -d` (http://localhost:6333)
- `.env`: GROQ_API_KEY (cevap üretimi için)

Kullanım:
  python stage3_naive_rag.py ingest              # varsayılan: samples/Yatirim_Fonu_Rehberi.pdf
  python stage3_naive_rag.py ingest --pdf baska.pdf
  python stage3_naive_rag.py ask "Sorunuz?"
  python stage3_naive_rag.py demo   # örnek PDF oluşturur, ingest + örnek soru

Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 boyut, stage2 ile uyumlu).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, PointStruct, VectorParams

load_dotenv(Path(__file__).resolve().parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
# Projede samples/ altındaki rehber PDF (ingest --pdf verilmezse kullanılır)
DEFAULT_INGEST_PDF = PROJECT_ROOT / "samples" / "Yatirim_Fonu_Rehberi.pdf"
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").strip()
COLLECTION = os.environ.get("NAIVE_RAG_COLLECTION", "naive_rag_pdf").strip()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
TOP_K = 5


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def extract_text_from_pdf(path: Path) -> str:
    import fitz

    doc = fitz.open(path)
    try:
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text())
        return "\n\n".join(parts)
    finally:
        doc.close()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap if end - overlap > start else end
    return chunks


def get_qdrant_client(url: str) -> QdrantClient:
    try:
        client = QdrantClient(url=url, timeout=15)
        client.get_collections()
        return client
    except Exception as exc:
        print(
            "Qdrant'a bağlanılamadı. Docker Desktop açık mı? Sonra:\n"
            f"  cd {PROJECT_ROOT}\n"
            "  docker compose up -d\n"
            f"Hata: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def ingest_pdf(
    pdf_path: Path,
    *,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection: str = COLLECTION,
    recreate: bool = True,
) -> int:
    if not pdf_path.is_file():
        print(f"Dosya bulunamadı: {pdf_path}", file=sys.stderr)
        return 1

    raw = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw)
    if not chunks:
        print("PDF'den metin çıkarılamadı veya boş.", file=sys.stderr)
        return 1

    from sentence_transformers import SentenceTransformer

    client = get_qdrant_client(qdrant_url)
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    try:
        existing = {c.name for c in client.get_collections().collections}
        if recreate:
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        elif collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    except UnexpectedResponse as exc:
        print(f"Koleksiyon hazırlanamadı: {exc}", file=sys.stderr)
        return 1

    source_name = pdf_path.name
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=len(chunks) > 20)
    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text": chunks[i],
                "chunk_index": i,
                "source": source_name,
            },
        )
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=collection, points=points)

    print(f"Kaynak: {pdf_path}")
    print(f"Koleksiyon: {collection} (cosine, boyut={dim})")
    print(f"Chunk sayısı: {len(chunks)}")
    return 0


def retrieve(
    query: str,
    *,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection: str = COLLECTION,
    source_filter: str | None = None,
    limit: int = TOP_K,
) -> list[tuple[float, str, dict]]:
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
    out: list[tuple[float, str, dict]] = []
    for h in res.points:
        payload = h.payload or {}
        text = str(payload.get("text", ""))
        out.append((float(h.score or 0.0), text, dict(payload)))
    return out


def answer_with_groq(query: str, contexts: list[str]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY tanımlı değil. .env içine ekleyin (console.groq.com).",
            file=sys.stderr,
        )
        sys.exit(1)

    from groq import Groq

    model = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
    ctx = "\n\n---\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(contexts))
    system = (
        "Sen yardımcı bir asistansın. Aşağıdaki bağlam parçalarına dayanarak kullanıcı "
        "sorusunu yanıtla. Bilgi bağlamda yoksa bunu açıkça söyle; uydurma."
    )
    user = f"Bağlam:\n{ctx}\n\nSoru: {query}"

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=512,
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def cmd_ingest(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf).resolve() if args.pdf else DEFAULT_INGEST_PDF.resolve()
    if not pdf_path.is_file():
        print(
            f"PDF bulunamadı: {pdf_path}\n"
            "Dosyayı samples/Yatirim_Fonu_Rehberi.pdf konumuna koyun veya "
            "`python stage3_naive_rag.py ingest --pdf <yol>` kullanın.",
            file=sys.stderr,
        )
        return 1
    return ingest_pdf(
        pdf_path,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        recreate=not args.no_recreate,
    )


def cmd_ask(args: argparse.Namespace) -> int:
    hits = retrieve(
        args.question,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        source_filter=args.source or None,
        limit=args.top_k,
    )
    if not hits:
        print("Retrieval sonuç vermedi. Önce `ingest` çalıştırın veya koleksiyonu kontrol edin.")
        return 1

    print("--- En yakın chunk'lar ---")
    for score, text, payload in hits:
        src = payload.get("source", "")
        print(f"  skor={score:.4f}  kaynak={src}")
        snippet = text.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:199] + "…"
        print(f"  «{snippet}»")
        print()

    contexts = [t for _, t, _ in hits if t.strip()]
    answer = answer_with_groq(args.question, contexts)
    print("--- LLM cevabı ---")
    print(answer)
    return 0


def write_demo_pdf(path: Path) -> None:
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    demo_text = (
        "İstanbul Boğazı, Karadeniz ile Marmara Denizi'ni birbirine bağlar. "
        "Boğazın iki yakasında yer alan şehir, tarihi ve kültürel açıdan önemlidir.\n\n"
        "Embedding vektörleri, metin parçalarını sabit boyutlu sayı dizilerine dönüştürür. "
        "Benzer anlamlı metinler, vektör uzayında birbirine yakın konumlanır.\n\n"
        "Naive RAG akışında önce belge parçalanır, sonra her parça embed edilir ve "
        "vektör veritabanına yazılır. Sorgu geldiğinde en yakın parçalar bulunur ve "
        "büyük dil modeline bağlam olarak verilir."
    )
    page.insert_text((72, 72), demo_text, fontsize=11)
    doc.save(path)
    doc.close()


def cmd_demo(args: argparse.Namespace) -> int:
    demo_pdf = (PROJECT_ROOT / "samples" / "demo_rag.pdf").resolve()
    write_demo_pdf(demo_pdf)
    print(f"Örnek PDF yazıldı: {demo_pdf}")
    rc = ingest_pdf(
        demo_pdf,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        recreate=True,
    )
    if rc != 0:
        return rc
    print()
    args.question = "Boğaz nereyi birbirine bağlar?"
    args.source = None
    args.top_k = TOP_K
    return cmd_ask(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Naive RAG: PDF ingest ve soru-cevap")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant HTTP adresi")
    p.add_argument(
        "--collection",
        default=COLLECTION,
        help="Qdrant koleksiyon adı (varsayılan: NAIVE_RAG_COLLECTION veya naive_rag_pdf)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="PDF'i chunk'layıp Qdrant'a yükle")
    pi.add_argument(
        "--pdf",
        default=None,
        metavar="YOL",
        help=(
            "PDF dosya yolu (belirtilmezse samples/Yatirim_Fonu_Rehberi.pdf kullanılır)"
        ),
    )
    pi.add_argument(
        "--no-recreate",
        action="store_true",
        help="Koleksiyonu silmeden üzerine yaz (önceki ingest ile aynı boyutta olmalı)",
    )
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="Sorgu: retrieval + Groq ile cevap")
    pa.add_argument("question", help="Soru metni")
    pa.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Kaç chunk getirilecek",
    )
    pa.add_argument(
        "--source",
        default="",
        help="Yalnızca bu kaynak dosya adına (payload.source) filtrele",
    )
    pa.set_defaults(func=cmd_ask)

    pd = sub.add_parser("demo", help="Örnek PDF oluştur, ingest et, örnek soru sor")
    pd.set_defaults(func=cmd_demo)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
