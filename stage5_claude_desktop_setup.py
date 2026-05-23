"""
Aşama 5 — Claude Desktop MCP entegrasyonu.

`%APPDATA%\\Claude\\claude_desktop_config.json` dosyasına `practice-usd-try`
sunucusunu ekler (mevcut preferences korunur).

Kullanım:
  python stage5_claude_desktop_setup.py           # config'e yaz
  python stage5_claude_desktop_setup.py --dry-run # sadece göster
  python stage5_claude_desktop_setup.py --remove  # config'ten kaldır

Sonra:
  1. Claude Desktop'ı tamamen kapat (system tray dahil)
  2. Yeniden aç
  3. Settings → Developer → MCP — practice-usd-try görünmeli
  4. Sohbette sor: "250 dolar kaç TL? usd_to_try tool'unu kullan."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SERVER_KEY = "practice-usd-try"
PROJECT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = PROJECT_DIR / "stage5_mcp_server.py"
CONFIG_PATH = Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"


def _server_entry() -> dict:
    return {
        "command": sys.executable,
        "args": [str(SERVER_SCRIPT.resolve())],
        "cwd": str(PROJECT_DIR),
    }


def _load_config() -> dict:
    if CONFIG_PATH.is_file():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"mcpServers": {}}


def _save_config(config: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def apply(*, dry_run: bool, remove: bool) -> None:
    if not SERVER_SCRIPT.is_file():
        raise SystemExit(f"Sunucu bulunamadı: {SERVER_SCRIPT}")

    config = _load_config()
    config.setdefault("mcpServers", {})

    entry = _server_entry()
    if remove:
        config["mcpServers"].pop(SERVER_KEY, None)
        action = "kaldırıldı"
    else:
        config["mcpServers"][SERVER_KEY] = entry
        action = "eklendi/güncellendi"

    payload = json.dumps(config, indent=2, ensure_ascii=False)
    print(f"Config: {CONFIG_PATH}")
    print(f"Sunucu: {SERVER_KEY} → {action}")
    print()
    print(json.dumps({SERVER_KEY: entry}, indent=2, ensure_ascii=False))

    if dry_run:
        print("\n(dry-run — dosya yazılmadı)")
        return

    _save_config(config)
    print(f"\nKaydedildi: {CONFIG_PATH}")
    print(
        "\nSonraki adımlar:\n"
        "  1. Claude Desktop'ı tamamen kapat\n"
        "  2. Yeniden aç\n"
        "  3. Settings → Developer → MCP bölümünde practice-usd-try'yi kontrol et\n"
        "  4. Sohbette: \"100 dolar kaç TL? usd_to_try tool'unu kullan.\""
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Claude Desktop MCP config kurulumu")
    parser.add_argument("--dry-run", action="store_true", help="Config yazmadan göster")
    parser.add_argument("--remove", action="store_true", help="practice-usd-try kaydını sil")
    args = parser.parse_args()
    apply(dry_run=args.dry_run, remove=args.remove)


if __name__ == "__main__":
    main()
