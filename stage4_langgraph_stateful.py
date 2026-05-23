"""
Aşama 4 — LangGraph ile durum taşıyan (stateful) ajan grafiği.

- **State**: mesaj geçmişi (`add_messages` reducer) + `refine_attempt` (toplamlayıcı).
- **Düğümler**: `call_model` (Groq) → `route_next` koşulu → gerekirse `nudge` ile kısa cevabı
  genişletmek için döngü → `END`.
- **Kalıcılık**: Varsayılan `SqliteSaver` — proje kökünde `.langgraph_stage4.sqlite`;
  aynı `--thread-id` ile sonraki `python ...` çağrıları önceki mesajları görür.
  İsteğe bağlı `--memory` ile yalnızca bellek içi `MemorySaver`.

Önkoşul: `.env` içinde `GROQ_API_KEY`

Kullanım:
  python stage4_langgraph_stateful.py "Python'da closure nedir?"
  python stage4_langgraph_stateful.py --thread-id demo1 "Merhaba"
  python stage4_langgraph_stateful.py --thread-id demo1 "Önceki mesajı özetle"
  python stage4_langgraph_stateful.py -v "kısa cevap ver"   # döngü adımlarını stderr'de göster
"""
from __future__ import annotations

import argparse
import operator
import os
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv(Path(__file__).resolve().parent / ".env")

MIN_ANSWER_CHARS = 80
MAX_REFINES = 2
CHECKPOINT_SQLITE = Path(__file__).resolve().parent / ".langgraph_stage4.sqlite"

SYSTEM = (
    "Sen yardımcı bir öğretmensin. Türkçe, net ve örnekli anlat. "
    "Çok kısa tek cümlelik cevaplardan kaçın; kavramı birkaç cümle ve gerekiyorsa madde işaretleriyle aç."
)


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


class GraphState(TypedDict):
    """Graf boyunca taşınan durum.

    `messages`: LangGraph'in `add_messages` reducer'ı ile birleştirilir (yeni mesajlar eklenir).
    `refine_attempt`: Her `nudge` düğümü +1 ekler — döngü sayısını sınırlamak için.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    refine_attempt: Annotated[int, operator.add]


def _build_llm() -> ChatGroq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY tanımlı değil (.env).", file=sys.stderr)
        sys.exit(1)
    model = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
    return ChatGroq(model=model, temperature=0.2, api_key=key)


def _last_ai_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""


def node_call_model(state: GraphState) -> dict[str, list[BaseMessage]]:
    llm = _build_llm()
    msgs: list[AnyMessage] = [SystemMessage(content=SYSTEM), *state["messages"]]
    ai: AIMessage = llm.invoke(msgs)
    return {"messages": [ai]}


def node_nudge(state: GraphState) -> dict[str, list[HumanMessage] | int]:
    return {
        "messages": [
            HumanMessage(
                content=(
                    f"Cevabın çok kısa görünüyor (en az {MIN_ANSWER_CHARS} karakter hedefle). "
                    "Aynı soruya daha ayrıntılı, örnekli ve yapılandırılmış şekilde yanıt ver."
                )
            )
        ],
        "refine_attempt": 1,
    }


def route_after_model(state: GraphState) -> Literal["nudge", "__end__"]:
    text = _last_ai_text(state["messages"])
    attempts = state.get("refine_attempt", 0)
    if attempts < MAX_REFINES and len(text.strip()) < MIN_ANSWER_CHARS:
        return "nudge"
    return "__end__"


def _last_ai_message(messages: list[AnyMessage]) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def _message_label(m: AnyMessage) -> str:
    t = type(m).__name__
    c = getattr(m, "content", "")
    if isinstance(c, str):
        preview = c.replace("\n", " ")[:60]
        return f"{t} len={len(c)} «{preview}{'…' if len(c) > 60 else ''}»"
    return f"{t} (non-text)"


def _describe_update(node: str, payload: object) -> str:
    if not isinstance(payload, dict):
        return str(payload)[:200]
    if node == "call_model":
        msgs = payload.get("messages") or []
        if msgs:
            return _message_label(msgs[-1])
        return "(mesaj yok)"
    if node == "nudge":
        msgs = payload.get("messages") or []
        ra = payload.get("refine_attempt")
        extra = f", refine_attempt += {ra}" if ra is not None else ""
        if msgs:
            return _message_label(msgs[-1]) + extra
        return extra or "(güncelleme yok)"
    return str(payload)[:200]


def run_invoke(
    graph,
    payload: dict,
    cfg: dict,
    *,
    verbose: bool,
) -> dict:
    """Tek `invoke` veya verbose ise `stream` ile adım adım stderr log."""
    if not verbose:
        return graph.invoke(payload, config=cfg)

    last_values: dict | None = None
    print("--- LangGraph stream (updates + values) ---", file=sys.stderr, flush=True)
    for chunk in graph.stream(
        payload,
        config=cfg,
        stream_mode=["updates", "values"],
    ):
        if not isinstance(chunk, tuple) or len(chunk) != 2:
            print(f"[?] {chunk!r}", file=sys.stderr, flush=True)
            continue
        mode, data = chunk
        if mode == "updates":
            for node_name, update in data.items():
                detail = _describe_update(node_name, update)
                print(
                    f"  [update] düğüm={node_name!r} → {detail}",
                    file=sys.stderr,
                    flush=True,
                )
        elif mode == "values":
            last_values = data
            msgs = data.get("messages") or []
            ra = data.get("refine_attempt", 0)
            tail = _message_label(msgs[-1]) if msgs else "(boş)"
            print(
                f"  [state] mesaj={len(msgs)} refine_attempt={ra} son={tail}",
                file=sys.stderr,
                flush=True,
            )
            if any(isinstance(m, AIMessage) for m in msgs):
                try:
                    nxt = route_after_model(data)
                    print(
                        f"  [yön] route_after_model -> {nxt!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"  [yön] (hesaplanamadı: {exc})",
                        file=sys.stderr,
                        flush=True,
                    )
    if last_values is None:
        raise RuntimeError("stream tamamlandı ancak values alınamadı")
    print("--- stream bitti ---\n", file=sys.stderr, flush=True)
    return last_values


def build_graph(*, use_memory: bool = False):
    g = StateGraph(GraphState)
    g.add_node("call_model", node_call_model)
    g.add_node("nudge", node_nudge)
    g.add_edge(START, "call_model")
    g.add_conditional_edges(
        "call_model",
        route_after_model,
        {"nudge": "nudge", "__end__": END},
    )
    g.add_edge("nudge", "call_model")
    if use_memory:
        checkpointer: MemorySaver | SqliteSaver = MemorySaver()
    else:
        conn = sqlite3.connect(str(CHECKPOINT_SQLITE), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    return g.compile(checkpointer=checkpointer)


def main() -> int:
    _ensure_utf8_stdio()
    p = argparse.ArgumentParser(description="LangGraph stateful örnek grafi")
    p.add_argument("question", nargs="?", default="Kısaca: asyncio event loop nedir?")
    p.add_argument(
        "--thread-id",
        "-t",
        default="default-thread",
        help="Checkpoint anahtarı (aynı id ile konuşma devam eder)",
    )
    p.add_argument(
        "--memory",
        action="store_true",
        help="Disk yerine yalnızca RAM'de checkpoint (süreç bitince silinir)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Döngü adımlarını stderr'de göster (stream: updates + values)",
    )
    args = p.parse_args()

    graph = build_graph(use_memory=args.memory)
    cfg = {"configurable": {"thread_id": args.thread_id}}

    out = run_invoke(
        graph,
        {"messages": [HumanMessage(content=args.question)], "refine_attempt": 0},
        cfg,
        verbose=args.verbose,
    )
    last_ai = _last_ai_message(out["messages"])
    if last_ai is None:
        print("Yanıt üretilemedi.", file=sys.stderr)
        return 1
    content = last_ai.content
    print(content if isinstance(content, str) else str(content))
    if out.get("refine_attempt", 0):
        print(f"\n(refine_attempt={out['refine_attempt']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
