"""
Aşama 1 — Görev 3: Function calling (tool use).

Sahte `get_weather` aracı tanımlanır; model `tool_calls` ile argümanları üretir,
sunucu fonksiyonu çalıştırır, sonuç `role: tool` mesajıyla modele geri verilir.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Belirtilen şehir için örnek hava özeti döndürür (demo verisi, gerçek API değil)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Şehir adı, örn. Ankara",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def _city_lookup_key(city: str) -> str:
    """Türkçe İ/I ile gelen şehir adlarını basitçe ASCII-benzeri anahtara çevirir."""
    s = city.strip().replace("İ", "i").replace("I", "i").lower()
    return "".join(ch for ch in s if ch.isalnum())


def get_weather(city: str) -> dict[str, Any]:
    """Sahte hava verisi — modelin çağırdığı araç bu fonksiyona bağlanır."""
    key = _city_lookup_key(city)
    samples: dict[str, dict[str, Any]] = {
        "ankara": {"temp_c": 18, "condition": "rüzgarlı, açık"},
        "istanbul": {"temp_c": 21, "condition": "parçalı bulutlu"},
        "izmir": {"temp_c": 24, "condition": "güneşli"},
    }
    row = samples.get(key, {"temp_c": 20, "condition": "bilinmeyen şehir — varsayılan özet"})
    return {"city": city.strip(), **row, "_note": "demo/mock"}


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY tanımlı değil. .env içine OPENAI_API_KEY=... ekleyin.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    model = "gpt-4o-mini"

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "Gerekirse get_weather aracını kullan; cevapları Türkçe ve kısa tut.",
        },
        {
            "role": "user",
            "content": "İstanbul ve Ankara için hava durumunu öğrenip tek paragrafta özetle.",
        },
    ]

    print("=== 1. tur: model araç seçebilir ===\n")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    assistant = resp.choices[0].message

    if not assistant.tool_calls:
        print("Model araç çağırmadı; doğrudan metin:")
        print(assistant.content or "(boş)")
        return

    print("[gözlem] assistant.tool_calls (modelin ürettiği çağrılar):\n")
    for tc in assistant.tool_calls:
        fn = tc.function
        print(f"  tool_call_id={tc.id}")
        print(f"  function.name={fn.name}")
        print(f"  function.arguments (ham JSON string)={fn.arguments!r}\n")

    messages.append(
        {
            "role": "assistant",
            "content": assistant.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in assistant.tool_calls
            ],
        }
    )

    for tc in assistant.tool_calls:
        name = tc.function.name
        raw_args = tc.function.arguments
        args = json.loads(raw_args) if raw_args else {}
        if name == "get_weather":
            city = args.get("city", "")
            print(f"[sunucu] get_weather({city!r}) çalıştırılıyor...\n")
            payload = get_weather(str(city))
            print(f"[sunucu] dönen dict: {payload}\n")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )
        else:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": f"bilinmeyen araç: {name}"}),
                }
            )

    print("=== 2. tur: araç çıktılarından nihai cevap ===\n")
    resp2 = client.chat.completions.create(model=model, messages=messages)
    final = resp2.choices[0].message.content
    print(final or "(boş cevap)")


if __name__ == "__main__":
    main()
