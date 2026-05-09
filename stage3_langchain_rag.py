"""
Aşama 3 — LangChain ile RAG (stage3_naive_rag ile aynı akış).

PDF yükle → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings (MiniLM)
→ Qdrant → Groq ile cevap.

Üç yanıt modu:
- **lcel**: LangChain Expression Language (Runnable / pipe) — varsayılan
- **retrieval-chain**: `create_retrieval_chain` + `create_stuff_documents_chain`
- **retrieval-qa**: `RetrievalQA.from_chain_type` (klasik zincir)

Önkoşullar: Docker Qdrant, `.env` içinde `GROQ_API_KEY`.

LangChain Qdrant kayıtları `page_content` payload kullanır; naive script `text` kullandığı için
varsayılan koleksiyon adı `langchain_rag_pdf` (NAIVE ile aynı koleksiyonda karışmaz).

Kullanım:
  python stage3_langchain_rag.py ingest
  python stage3_langchain_rag.py ask "Soru?"
  python stage3_langchain_rag.py ask "Soru?" --mode retrieval-chain
  python stage3_langchain_rag.py ask "Soru?" --mode retrieval-qa
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INGEST_PDF = PROJECT_ROOT / "samples" / "Yatirim_Fonu_Rehberi.pdf"
DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").strip()
# LangChain Qdrant varsayılan koleksiyon (NAIVE_RAG_COLLECTION ile karıştırmamak için ayrı ad)
COLLECTION = os.environ.get("LANGCHAIN_RAG_COLLECTION", "langchain_rag_pdf").strip()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
TOP_K = 5

SYSTEM_RAG = (
    "Sen yardımcı bir asistansın. Verilen bağlam parçalarına dayanarak soruyu yanıtla. "
    "Bilgi bağlamda yoksa bunu açıkça söyle; uydurma."
)

def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _tokenizers_parallelism_off() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_embeddings():
    _tokenizers_parallelism_off()
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": False},
    )


def build_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY tanımlı değil. .env içine ekleyin (console.groq.com).",
            file=sys.stderr,
        )
        sys.exit(1)

    from langchain_groq import ChatGroq

    model = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
    return ChatGroq(model=model, temperature=0.2, api_key=api_key)


def load_pdf_documents(pdf_path: Path):
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def ingest_to_qdrant(
    pdf_path: Path,
    *,
    qdrant_url: str,
    collection: str,
    force_recreate: bool,
):
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import Distance, VectorParams

    if not pdf_path.is_file():
        print(f"Dosya bulunamadı: {pdf_path}", file=sys.stderr)
        return 1

    embeddings = build_embeddings()
    raw_docs = load_pdf_documents(pdf_path)
    docs = split_documents(raw_docs)
    if not docs:
        print("PDF'den metin çıkarılamadı veya chunk üretilemedi.", file=sys.stderr)
        return 1

    for d in docs:
        d.metadata.setdefault("source", pdf_path.name)

    # Koleksiyonu qdrant-client ile yönetiyoruz; vektör katmanı olarak langchain-qdrant
    # (güncel qdrant-client'ta kaldırılan client.search() yüzünden langchain_community Qdrant kullanılmıyor).

    client = QdrantClient(url=qdrant_url, timeout=30)
    try:
        client.get_collections()
    except Exception as exc:
        print(
            "Qdrant'a bağlanılamadı. `docker compose up -d` ile başlatın.\n"
            f"Hata: {exc}",
            file=sys.stderr,
        )
        return 1

    dim = len(embeddings.embed_query("boyut"))
    try:
        existing = {c.name for c in client.get_collections().collections}
        if force_recreate:
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

    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )
    store.add_documents(docs)

    print(f"Kaynak: {pdf_path}")
    print(f"Koleksiyon: {collection}")
    print(f"LangChain Document sayısı (chunk): {len(docs)}")
    return 0


def connect_vectorstore(qdrant_url: str, collection: str):
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    embeddings = build_embeddings()
    client = QdrantClient(url=qdrant_url, timeout=15)
    try:
        client.get_collections()
    except Exception as exc:
        print(
            "Qdrant'a bağlanılamadı. `docker compose up -d` ile başlatın.\n"
            f"Hata: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def chain_lcel(vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_RAG),
            ("human", "Bağlam:\n{context}\n\nSoru: {question}"),
        ]
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def chain_retrieval(vectorstore, llm):
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_RAG),
            ("human", "Bağlam:\n{context}\n\nSoru: {input}"),
        ]
    )
    combine_docs = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs)


def chain_retrieval_qa(vectorstore, llm):
    from langchain_classic.chains import RetrievalQA
    from langchain_core.prompts import PromptTemplate

    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    pt = PromptTemplate(
        template=(
            SYSTEM_RAG
            + "\n\n"
            "Bağlam:\n{context}\n\n"
            "Soru: {question}\n\n"
            "Yanıt:"
        ),
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": pt},
        return_source_documents=True,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf).resolve() if args.pdf else DEFAULT_INGEST_PDF.resolve()
    return ingest_to_qdrant(
        pdf_path,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        force_recreate=not args.no_recreate,
    )


def cmd_ask(args: argparse.Namespace) -> int:
    vs = connect_vectorstore(args.qdrant_url, args.collection)
    llm = build_llm()

    mode = args.mode.strip().lower()
    question = args.question

    if mode == "lcel":
        chain = chain_lcel(vs, llm)
        answer = chain.invoke(question)
        print(answer)
        return 0

    if mode == "retrieval-chain":
        chain = chain_retrieval(vs, llm)
        out = chain.invoke({"input": question})
        print(out.get("answer", out))
        return 0

    if mode == "retrieval-qa":
        qa = chain_retrieval_qa(vs, llm)
        out = qa.invoke({"query": question})
        print(out["result"])
        return 0

    print(f"Bilinmeyen mod: {mode}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LangChain RAG: Qdrant + Groq")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument(
        "--collection",
        default=COLLECTION,
        help="Qdrant koleksiyonu (varsayılan LANGCHAIN_RAG_COLLECTION veya langchain_rag_pdf)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="PDF → chunk → embed → Qdrant")
    pi.add_argument(
        "--pdf",
        default=None,
        metavar="YOL",
        help="PDF (yoksa samples/Yatirim_Fonu_Rehberi.pdf)",
    )
    pi.add_argument(
        "--no-recreate",
        action="store_true",
        help="Koleksiyonu silmeden üzerine ekle (uyumlu boyut gerekir)",
    )
    pi.set_defaults(func=cmd_ingest)

    pa = sub.add_parser("ask", help="Retrieval + LLM")
    pa.add_argument("question")
    pa.add_argument(
        "--mode",
        choices=("lcel", "retrieval-chain", "retrieval-qa"),
        default="lcel",
        help="lcel | retrieval-chain (create_retrieval_chain) | retrieval-qa (RetrievalQA)",
    )
    pa.set_defaults(func=cmd_ask)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    _ensure_utf8_stdio()
    main()
