"""
Aşama 5 — Tek tool'lu MCP sunucusu (resmi Python `mcp` SDK / FastMCP).

Sunucu tek bir tool sunar: `usd_to_try` — Frankfurter API ile güncel USD/TRY kurunu
alır ve verilen USD tutarını TL'ye çevirir.

Transport: stdio (Claude Desktop, Cursor vb. host'lar bu modda başlatır).

Kullanım:
  python stage5_mcp_server.py              # stdio modunda dinler (çıktı YOK — normal)
  python stage5_mcp_server.py --self-test  # terminalde sonuç görmek için bunu kullan

Not: stdio modunda stdout MCP protokolüne ayrılır; terminalde sonuç görmek için
--self-test kullanın veya stage5_mcp_client.py çalıştırın.

Claude Desktop entegrasyonu:
  python stage5_claude_desktop_setup.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.error
import urllib.request
from datetime import timedelta

from mcp.server.fastmcp import FastMCP

_HTTP_UA = "Mozilla/5.0 (compatible; llm-rag-practice-stage5/1.0)"

mcp = FastMCP("practice-usd-try")


def _fetch_usd_try_rate() -> tuple[float, str]:
    """Frankfurter'dan güncel USD/TRY kurunu al."""
    req = urllib.request.Request(
        "https://api.frankfurter.app/latest?from=USD&to=TRY",
        headers={"User-Agent": _HTTP_UA},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    rate = float(data["rates"]["TRY"])
    date = str(data.get("date", "unknown"))
    return rate, date


@mcp.tool()
def usd_to_try(amount: float) -> str:
    """Verilen USD tutarını güncel kur ile Türk Lirasına çevirir.

    Args:
        amount: Dönüştürülecek ABD doları miktarı (ör. 100.0)
    """
    if amount < 0:
        raise ValueError("amount negatif olamaz")
    try:
        rate, date = _fetch_usd_try_rate()
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        return f"Hata: kur alınamadı ({exc})"

    total_try = amount * rate
    return (
        f"{amount:.2f} USD = {total_try:,.2f} TRY "
        f"(1 USD = {rate:.4f} TRY, tarih: {date}, kaynak: frankfurter.app)"
    )


async def _self_test() -> None:
    """Sunucuyu bellek içi transport ile başlatıp tool'u doğrula."""
    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(
        mcp,
        read_timeout_seconds=timedelta(seconds=30),
    ) as session:
        tools = await session.list_tools()
        names = [t.name for t in tools.tools]
        print(f"Tools: {names}")
        assert "usd_to_try" in names, "usd_to_try tool bulunamadı"

        result = await session.call_tool("usd_to_try", {"amount": 100.0})
        text_blocks = [b.text for b in result.content if hasattr(b, "text")]
        print("usd_to_try(100):", text_blocks[0] if text_blocks else result.content)
        print("Self-test OK.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Tek tool'lu MCP sunucusu (usd_to_try)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Bellek içi istemci ile tool çağrısını dene (stdio başlatmaz)",
    )
    args = parser.parse_args()

    if args.self_test:
        asyncio.run(_self_test())
    else:
        print(
            "MCP sunucusu stdio modunda bekliyor (tool: usd_to_try).\n"
            "Bu modda terminalde çıktı gelmez — host (Cursor/Claude Desktop) stdin'e "
            "JSON-RPC gönderir.\n"
            "Sonucu görmek için: python stage5_mcp_server.py --self-test\n"
            "Durdurmak için: Ctrl+C",
            file=sys.stderr,
        )
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
