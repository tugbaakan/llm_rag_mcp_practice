"""
Aşama 4 — LangChain `create_agent` ile ReAct benzeri ajan (düşün → araç çağır → tekrar).

`spot_reference_try` (Frankfurter + yedek open.er-api USD/TRY; Binance PAXG ≈ troy ons altın),
`web_search` (DuckDuckGo, tr-tr) ve `calculator` ile portföyün TL karşılığını hesaplar.

Örnek senaryo (görev metni):
  3 cumhuriyet altını, 5 gr altın, 100 USD — günün kurları / fiyatları ile toplam değer.

Önkoşullar:
  - `.env` içinde `GROQ_API_KEY`
  - `pip install -r requirements.txt` (DuckDuckGo için `ddgs` paketi gerekir)

Kullanım:
  python stage4_react_agent.py
  python stage4_react_agent.py --question "Özel soru metni"
  python stage4_react_agent.py --verbose
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

_HTTP_UA = "Mozilla/5.0 (compatible; llm-rag-practice-stage4/1.0)"

DEFAULT_QUESTION = (
    "Elimde 3 cumhuriyet altını, 5 gram altın ve 100 ABD doları banknot var. "
    "Önce spot_reference_try ile güncel USD/TRY ve uluslararası altın referansını al; "
    "cumhuriyet altını ve piyasa gram fiyatı için web_search kullan (Türkçe sorgular). "
    "Tüm çarpma ve toplamayı calculator ile yap. "
    "Sonucu Türk Lirası cinsinden net toplam ve sayı kaynaklarını kısaca özetle."
)

SYSTEM_PROMPT = (
    "Sen planlı bir finans asistanısın. Döviz ve uluslararası altın referansı için "
    "mutlaka spot_reference_try kullan (API; kurları uydurma). "
    "Cumhuriyet altını ve güncel kuyum/banka gram satış fiyatı için web_search kullan; "
    "arama sonucunda net rakam yoksa spot_reference_try'deki gram başına TL tahminini "
    "açıkça 'yaklaşık / uluslararası referans' diye belirt. "
    "İşlemleri calculator ile yap. Yanıtı Türkçe ver."
)


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _eval_math(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        ops: dict[type[ast.operator], type] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
        }
        t = type(node.op)
        if t not in ops:
            raise ValueError(f"desteklenmeyen ikili islem: {type(node.op).__name__}")
        return float(ops[t](_eval_math(node.left), _eval_math(node.right)))
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_math(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _eval_math(node.operand)
        raise ValueError(f"desteklenmeyen tek terimli islem: {type(node.op).__name__}")
    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("fonksiyon cagrisinda keyword arguman yok")
        if not isinstance(node.func, ast.Name):
            raise ValueError("sadece round ve abs desteklenir")
        fn = node.func.id
        if fn == "round":
            if len(node.args) == 1:
                return float(round(_eval_math(node.args[0])))
            if len(node.args) == 2:
                return float(round(_eval_math(node.args[0]), int(_eval_math(node.args[1]))))
            raise ValueError("round 1 veya 2 arguman alir")
        if fn == "abs" and len(node.args) == 1:
            return float(abs(_eval_math(node.args[0])))
        raise ValueError("sadece round ve abs desteklenir")
    raise ValueError(f"desteklenmeyen ifade: {type(node).__name__}")


def build_calculator_tool():
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Güvenli aritmetik: sayılar, + - * / ** // %, parantez, round(x), round(x,n), abs(x).

        Ondalık için nokta kullan; Türkçe tek ondalık virgülü (round/abs yoksa) noktaya çeviririz.
        """
        raw = expression.strip()
        if re.search(r"\bround\b|\babs\b", raw, re.IGNORECASE):
            expr = raw
        else:
            expr = raw.replace(",", ".")
        try:
            tree = ast.parse(expr, mode="eval")
            return str(_eval_math(tree.body))
        except SyntaxError as e:
            return f"HATA (sozdizimi): {e}. Ornek: (3*15000)+(100*45.36) veya round(1234.56,2)"
        except ValueError as e:
            return f"HATA (ifade): {e}. Ornek: sadece sayilar, +-*/**, round(), abs(); degisken yok."
        except ZeroDivisionError:
            return "HATA: sifira bolme."

    return calculator


def _http_get_json(url: str, timeout: float = 25.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _HTTP_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def build_spot_reference_tool():
    """Frankfurter (ECB) USD/TRY + Binance PAXG — DuckDuckGo'dan bağımsız canlı rakamlar."""
    from langchain_core.tools import tool

    troy_oz_grams = 31.1034768

    @tool
    def spot_reference_try() -> str:
        """Güncel USD/TRY ve yaklaşık spot altın (PAXG ≈ 1 troy ons altın) referansı, TL'ye çevrilmiş.

        Cumhuriyet altını adet fiyatı veya Türkiye kuyum spread'i içermez; bunlar için web_search gerekir.
        """
        errors: list[str] = []
        usd_try: float | None = None
        rate_date = ""
        try:
            data = _http_get_json("https://api.frankfurter.app/latest?from=USD&to=TRY")
            usd_try = float(data["rates"]["TRY"])
            rate_date = str(data.get("date", ""))
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, TypeError) as e:
            errors.append(f"frankfurter: {e}")
        if usd_try is None:
            try:
                data = _http_get_json("https://open.er-api.com/v6/latest/USD")
                if data.get("result") == "success":
                    usd_try = float(data["rates"]["TRY"])
                    rate_date = "open.er-api.com"
            except (urllib.error.URLError, KeyError, ValueError, TypeError) as e:
                errors.append(f"open.er-api: {e}")

        paxg_usd: float | None = None
        try:
            row = _http_get_json("https://api.binance.com/api/v3/ticker/price?symbol=PAXGUSDT")
            paxg_usd = float(row["price"])
        except (urllib.error.URLError, KeyError, ValueError, TypeError) as e:
            errors.append(f"binance_paxg: {e}")

        if usd_try is None:
            return "USD/TRY alınamadı. " + "; ".join(errors)

        lines = [
            f"USD/TRY (ECB tabanlı Frankfurter veya yedek API): {usd_try} (tarih/kaynak: {rate_date or 'bilinmiyor'})",
            "Not: TCMB efektif / serbest piyasa ile küçük fark olabilir.",
        ]
        if paxg_usd is not None:
            usd_per_g = paxg_usd / troy_oz_grams
            try_per_g = usd_per_g * usd_try
            lines.append(
                f"PAXG USDT (~1 troy ons altın): {paxg_usd} USD; "
                f"≈ {usd_per_g:.4f} USD/gram; ≈ {try_per_g:.2f} TRY/gram (referans, kuyum fiyatı değil)."
            )
        else:
            lines.append("PAXG/USD alınamadı; gram altın TL hesabı için web_search gerekir. " + "; ".join(errors))
        return "\n".join(lines)

    return spot_reference_try


def build_web_search_tool():
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

    wrapper = DuckDuckGoSearchAPIWrapper(
        region="tr-tr",
        max_results=10,
        time="m",
        backend="lite",
    )
    return DuckDuckGoSearchRun(api_wrapper=wrapper)


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
    return ChatGroq(model=model, temperature=0.1, api_key=api_key)


def build_agent():
    from langchain.agents import create_agent

    llm = build_llm()
    tools = [build_spot_reference_tool(), build_web_search_tool(), build_calculator_tool()]
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def run_agent(question: str, *, verbose: bool) -> str:
    from langchain_core.messages import HumanMessage

    agent = build_agent()
    if not verbose:
        out = agent.invoke({"messages": [HumanMessage(content=question)]})
        last = out["messages"][-1]
        return getattr(last, "content", str(last))

    chunks: list[str] = []
    for part in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values",
    ):
        msgs = part.get("messages") or []
        if not msgs:
            continue
        m = msgs[-1]
        label = type(m).__name__
        content = getattr(m, "content", None)
        tool_calls = getattr(m, "tool_calls", None)
        line = f"[{label}]"
        if tool_calls:
            line += f" tool_calls={tool_calls}"
        if content:
            line += f" {content}"
        print(line, flush=True)
        if label == "AIMessage" and content:
            chunks.append(str(content))
    return chunks[-1] if chunks else ""


def main() -> int:
    _ensure_utf8_stdio()
    p = argparse.ArgumentParser(description="ReAct tarzı ajan: web + hesap makinesi")
    p.add_argument(
        "--question",
        "-q",
        default=DEFAULT_QUESTION,
        help="Modele gönderilecek görev metni",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Ara mesajları ve tool çağrılarını yazdır",
    )
    args = p.parse_args()

    try:
        answer = run_agent(args.question, verbose=args.verbose)
    except ImportError as e:
        print(
            f"Eksik paket: {e}\n"
            "DuckDuckGo için: pip install -U ddgs\n"
            "Tüm bağımlılıklar: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 1

    if not args.verbose:
        print(answer)
    elif answer:
        print("\n--- Özet cevap ---\n", answer, sep="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
