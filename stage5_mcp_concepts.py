"""
Aşama 5 — Model Context Protocol (MCP) kavramları.

Spec kaynağı: https://modelcontextprotocol.io/specification/latest
Mimari özeti: https://modelcontextprotocol.io/docs/learn

MCP, LLM uygulamalarının (host) dış veri kaynakları ve yeteneklerle konuşması için
JSON-RPC 2.0 tabanlı açık bir protokoldür. USB-C benzetmesiyle: her araç kendi
entegrasyonunu yazmak yerine standart bir sokete takılır.

Katmanlar
---------
- Transport: stdio (lokal süreç) veya Streamable HTTP (uzak sunucu)
- Data layer: lifecycle, primitives (tools/resources/prompts), bildirimler

Katılımcılar
------------
- Host: LLM uygulaması (Cursor, Claude Desktop, VS Code …)
- Client: Host içinde, tek bir sunucuya özel bağlantı yönetir
- Server: Context ve yetenek sağlayan program (lokal veya remote)

Sunucu primitives (3 temel kavram)
----------------------------------
1. Tools   — Modelin ÇAĞIRABİLECEĞİ fonksiyonlar (eylem)
2. Resources — Modele OKUNABİLİR bağlam verisi (dosya, şema, API yanıtı …)
3. Prompts — Yeniden kullanılabilir mesaj şablonları (few-shot, workflow)

İstemci primitives (sunucunun host'tan isteyebileceği)
------------------------------------------------------
- Sampling: Sunucu, host LLM'den completion ister (model bağımsız kalır)
- Elicitation: Kullanıcıdan ek bilgi/onay ister
- Logging: Debug logları client'a gönderir

Yaşam döngüsü
-------------
initialize → capability negotiation → notifications/initialized → */list → kullanım

Kullanım:
  python stage5_mcp_concepts.py
  python stage5_mcp_concepts.py --section tools
  python stage5_mcp_concepts.py --inspect-cursor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Özet içerik — spec'ten derlenmiş referans
# ---------------------------------------------------------------------------

ARCHITECTURE = dedent("""
    +---------------------------------------------------------+
    |  MCP Host (Cursor, Claude Desktop, ...)                 |
    |  +-------------+  +-------------+  +-------------+    |
    |  | MCP Client 1|  | MCP Client 2|  | MCP Client 3|    |
    |  +------+------+  +------+------+  +------+------+    |
    +---------+----------------+----------------+-----------+
              | stdio/HTTP     |                |
         +----v----+     +-----v-----+   +-----v-----+
         | Server A|     | Server B  |   | Server C  |
         | (lokal) |     | (lokal)   |   | (remote)  |
         +---------+     +-----------+   +-----------+
""")

TOOLS = {
    "rol": "Modelin çalıştırabileceği eylemler (API çağrısı, DB sorgusu, hesaplama …)",
    "keşif": "tools/list  →  sunucudaki tool listesi (pagination destekli)",
    "çağrı": "tools/call  →  name + arguments; JSON Schema ile doğrulanır",
    "bildirim": "notifications/tools/list_changed  (listChanged capability varsa)",
    "alanlar": ["name", "title", "description", "inputSchema", "outputSchema (opsiyonel)"],
    "örnek_list": {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
    },
    "örnek_call": {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "weather_current",
            "arguments": {"location": "Istanbul", "units": "metric"},
        },
    },
}

RESOURCES = {
    "rol": "Modele okunabilir bağlam — dosya içeriği, DB şeması, uygulama durumu …",
    "kimlik": "Her resource bir URI ile tanımlanır (file://, https://, özel scheme)",
    "keşif": "resources/list  +  resources/templates/list (parametreli URI şablonları)",
    "okuma": "resources/read  →  text veya base64 blob içerik",
    "abonelik": "resources/subscribe + notifications/resources/updated (opsiyonel)",
    "annotations": {
        "audience": '["user", "assistant"] — kime yönelik',
        "priority": "0.0–1.0 — bağlama dahil etme önceliği",
        "lastModified": "ISO 8601 zaman damgası",
    },
    "örnek_read": {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/read",
        "params": {"uri": "file:///project/README.md"},
    },
}

PROMPTS = {
    "rol": "Kullanıcı veya host için hazır mesaj şablonları (few-shot, code review …)",
    "keşif": "prompts/list  →  name, description, arguments listesi",
    "kullanım": "prompts/get  →  name + arguments; render edilmiş messages[] döner",
    "bildirim": "notifications/prompts/list_changed  (listChanged capability varsa)",
    "fark_tooldan": (
        "Tool = model karar verip çağırır (agentic). "
        "Prompt = kullanıcı/host bilinçli seçer (UI'da slash command gibi)."
    ),
    "örnek_get": {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "prompts/get",
        "params": {
            "name": "code_review",
            "arguments": {"code": "def hello(): print('world')"},
        },
    },
}

COMPARISON = dedent("""
    +------------+--------------------------+-----------------------------+
    | Primitive  | Kim tetikler?            | Tipik kullanim              |
    +------------+--------------------------+-----------------------------+
    | Tools      | LLM (agent)              | Hava durumu, SQL, dosya yaz |
    | Resources  | Host / kullanici / LLM   | README, sema, log dosyasi   |
    | Prompts    | Kullanici / host         | "Kodu incele", git commit   |
    +------------+--------------------------+-----------------------------+
""")

INITIALIZE = {
    "client": {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"elicitation": {}},
            "clientInfo": {"name": "example-client", "version": "1.0.0"},
        },
    },
    "server_yanıt": {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": True},
                "prompts": {"listChanged": True},
            },
            "serverInfo": {"name": "example-server", "version": "1.0.0"},
        },
    },
}


def _print_json(label: str, obj: object) -> None:
    print(f"\n  {label}:")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def print_section(name: str) -> None:
    """Tek bir bölümü yazdır."""
    sections = {
        "architecture": lambda: print(ARCHITECTURE),
        "tools": lambda: (_print_block("TOOLS", TOOLS)),
        "resources": lambda: (_print_block("RESOURCES", RESOURCES)),
        "prompts": lambda: (_print_block("PROMPTS", PROMPTS)),
        "comparison": lambda: print(COMPARISON),
        "initialize": lambda: (
            print("\n=== initialize (capability negotiation) ==="),
            _print_json("Client → Server", INITIALIZE["client"]),
            _print_json("Server → Client", INITIALIZE["server_yanıt"]),
        ),
    }
    fn = sections.get(name)
    if fn is None:
        print(f"Bilinmeyen bölüm: {name}. Seçenekler: {', '.join(sections)}")
        sys.exit(1)
    fn()


def _print_block(title: str, data: dict) -> None:
    print(f"\n=== {title} ===")
    for key, val in data.items():
        if key.startswith("örnek"):
            _print_json(key, val)
        elif isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        elif isinstance(val, list):
            print(f"  {key}: {', '.join(val)}")
        else:
            print(f"  {key}: {val}")


def print_all() -> None:
    print("=" * 60)
    print("Model Context Protocol — Özet Referans")
    print("Spec: https://modelcontextprotocol.io/specification/latest")
    print("=" * 60)
    print_section("architecture")
    print_section("comparison")
    print_section("tools")
    print_section("resources")
    print_section("prompts")
    print_section("initialize")
    print("\nSonraki adımlar (task-list): MCP sunucusu yaz → istemci yaz → Claude Desktop.")


def inspect_cursor_mcps() -> None:
    """Cursor'ın bu workspace için kaydettiği MCP sunucu meta verisini listele."""
    # Cursor, MCP tool şemalarını proje mcps/ altında tutar (host tarafı cache).
    candidates = [
        Path(__file__).resolve().parent.parent.parent
        / ".cursor"
        / "projects"
        / "c-Users-Nuevohp-disc-folder-kisiselBelgeler-MISC-llm-rag-mcp-practice"
        / "mcps",
        Path.home()
        / ".cursor"
        / "projects"
        / "c-Users-Nuevohp-disc-folder-kisiselBelgeler-MISC-llm-rag-mcp-practice"
        / "mcps",
    ]
    mcps_dir = next((p for p in candidates if p.is_dir()), None)
    if mcps_dir is None:
        print("Cursor mcps/ klasörü bulunamadı.")
        return

    print(f"\n=== Cursor MCP sunucuları ({mcps_dir}) ===")
    for server_dir in sorted(mcps_dir.iterdir()):
        if not server_dir.is_dir():
            continue
        meta = server_dir / "SERVER_METADATA.json"
        print(f"\n  Sunucu: {server_dir.name}")
        if meta.is_file():
            print(f"    metadata: {meta.read_text(encoding='utf-8').strip()}")

        tools_dir = server_dir / "tools"
        if tools_dir.is_dir():
            for tool_file in sorted(tools_dir.glob("*.json")):
                try:
                    schema = json.loads(tool_file.read_text(encoding="utf-8"))
                    name = schema.get("name", tool_file.stem)
                    desc = schema.get("description", "")[:80]
                    print(f"    tool: {name} — {desc}")
                except json.JSONDecodeError:
                    print(f"    tool: {tool_file.name} (JSON okunamadı)")
        else:
            status = server_dir / "STATUS.md"
            if status.is_file():
                first_line = status.read_text(encoding="utf-8").splitlines()[0]
                print(f"    durum: {first_line}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="MCP kavramları referansı")
    parser.add_argument(
        "--section",
        choices=["architecture", "tools", "resources", "prompts", "comparison", "initialize"],
        help="Tek bir bölüm yazdır",
    )
    parser.add_argument(
        "--inspect-cursor",
        action="store_true",
        help="Cursor'ın kayıtlı MCP sunucu/tool tanımlarını listele",
    )
    args = parser.parse_args()

    if args.section:
        print_section(args.section)
    else:
        print_all()

    if args.inspect_cursor:
        inspect_cursor_mcps()


if __name__ == "__main__":
    main()
