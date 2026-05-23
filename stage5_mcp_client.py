"""
Aşama 5 — MCP istemcisi: `stage5_mcp_server.py` sunucusuna stdio ile bağlanır
ve tool çağrısını manuel tetikler.

Akış:
  1. Sunucu sürecini spawn et (stdio transport)
  2. initialize → capability negotiation
  3. tools/list → mevcut tool'ları keşfet
  4. tools/call → usd_to_try(amount=...) çalıştır

Kullanım:
  python stage5_mcp_client.py
  python stage5_mcp_client.py --amount 250
  python stage5_mcp_client.py --tool usd_to_try --amount 50
  python stage5_mcp_client.py --list-only
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DEFAULT_SERVER = Path(__file__).resolve().parent / "stage5_mcp_server.py"
DEFAULT_TOOL = "usd_to_try"
DEFAULT_AMOUNT = 100.0


def _tool_result_text(content: list) -> str:
    parts: list[str] = []
    for block in content:
        if hasattr(block, "text") and block.text:
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


async def run_client(
    *,
    server_path: Path,
    tool_name: str,
    amount: float,
    list_only: bool,
) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        cwd=str(server_path.parent),
    )

    print(f"Sunucu başlatılıyor: {sys.executable} {server_path.name}")
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            print(f"Bağlandı: {init.serverInfo.name} v{init.serverInfo.version}")

            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            print(f"Tool'lar: {tool_names}")

            if list_only:
                for tool in tools_result.tools:
                    print(f"  - {tool.name}: {tool.description or '(açıklama yok)'}")
                return

            if tool_name not in tool_names:
                raise SystemExit(f"Tool bulunamadı: {tool_name!r}. Mevcut: {tool_names}")

            print(f"\nÇağrılıyor: {tool_name}(amount={amount})")
            result = await session.call_tool(tool_name, {"amount": amount})
            print("\nSonuç:")
            print(_tool_result_text(result.content))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="MCP istemcisi — stage5 sunucusuna bağlan")
    parser.add_argument(
        "--server",
        type=Path,
        default=DEFAULT_SERVER,
        help="Bağlanılacak MCP sunucu script'i",
    )
    parser.add_argument(
        "--tool",
        default=DEFAULT_TOOL,
        help=f"Çağrılacak tool adı (varsayılan: {DEFAULT_TOOL})",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=DEFAULT_AMOUNT,
        help=f"usd_to_try için USD miktarı (varsayılan: {DEFAULT_AMOUNT})",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Sadece tools/list yap, tool çağırma",
    )
    args = parser.parse_args()

    if not args.server.is_file():
        raise SystemExit(f"Sunucu bulunamadı: {args.server}")

    asyncio.run(
        run_client(
            server_path=args.server.resolve(),
            tool_name=args.tool,
            amount=args.amount,
            list_only=args.list_only,
        )
    )


if __name__ == "__main__":
    main()
