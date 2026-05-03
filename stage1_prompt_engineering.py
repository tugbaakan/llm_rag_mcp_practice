"""
Aşama 1 — Görev 4: Prompt engineering — 3 senaryo.

1) Yalnızca system prompt ile kısıtlar ve rol
2) Few-shot: örnek kullanıcı/asistan çiftleri
3) Chain-of-thought: önce adım adım düşünme, sonra kısa cevap
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "gpt-4o-mini"


def _configure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}\n")


def scenario_system_prompt_only(client: OpenAI) -> None:
    """Senaryo A: Örnek yok; kurallar ve rol tamamen system prompt’ta."""
    _banner("Senaryo 1 — System prompt (few-shot yok)")
    messages = [
        {
            "role": "system",
            "content": (
                "Sen bir müşteri mesajı sınıflandırıcısısın. Her kullanıcı mesajını "
                "yalnızca şu etiketlerden biriyle yanıtla: SIPARIS, SIKAYET, BILGI.\n"
                "Kurallar:\n"
                "- Sipariş, teslimat veya ürün satın alma niyeti → SIPARIS\n"
                "- Şikayet, iade, kırık ürün, memnuniyetsizlik → SIKAYET\n"
                "- Genel soru, fiyat/özellik sorusu, bilgi talebi → BILGI\n"
                "Yanıtta başka kelime veya açıklama yazma; sadece tek etiket."
            ),
        },
        {
            "role": "user",
            "content": "Kargom hâlâ gelmedi, çok sinirliyim.",
        },
    ]
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=20,
        temperature=0,
    )
    print(r.choices[0].message.content or "(boş)")


def scenario_few_shot(client: OpenAI) -> None:
    """Senaryo B: Aynı görev; davranış örnek diyaloglarla verilir."""
    _banner("Senaryo 2 — Few-shot örnekleri")
    messages = [
        {
            "role": "system",
            "content": (
                "Aşağıdaki örneklerle aynı biçimde yanıt ver: yalnızca SIPARIS, SIKAYET "
                "veya BILGI kelimelerinden biri; başka kelime veya noktalama yok."
            ),
        },
        {"role": "user", "content": "Bu üründen iki adet gönderir misiniz?"},
        {"role": "assistant", "content": "SIPARIS"},
        {"role": "user", "content": "Paket ezilmiş geldi, paramı istiyorum."},
        {"role": "assistant", "content": "SIKAYET"},
        {"role": "user", "content": "Garanti süresi kaç yıl?"},
        {"role": "assistant", "content": "BILGI"},
        {
            "role": "user",
            "content": "Bu modelin pil ömrü kaç saat sürer?",
        },
    ]
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=20,
        temperature=0,
    )
    print(r.choices[0].message.content or "(boş)")


def scenario_chain_of_thought(client: OpenAI) -> None:
    """Senaryo C: Önce düşünce zinciri, sonra nihai cevap (etiketlerle ayrıştırılmış)."""
    _banner("Senaryo 3 — Chain-of-thought (adım adım, sonra özet)")
    messages = [
        {
            "role": "system",
            "content": (
                "Önce problemi adım adım çöz. Tüm ara çıkarımları <dusunce> ve </dusunce> "
                "etiketleri arasına yaz.\n"
                "En sonda yalnızca tek sayıyı <cevap> ve </cevap> arasına koy; "
                "başka metin ekleme."
            ),
        },
        {
            "role": "user",
            "content": (
                "Bir tezgahta 5 kırmızı, 4 mavi blok var. 2 kırmızı ve 1 mavi blok "
                "alındı. Sonra 3 yeşil blok eklendi. Tezgahta toplam kaç blok var?"
            ),
        },
    ]
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0,
    )
    print(r.choices[0].message.content or "(boş)")


def main() -> None:
    _configure_stdio_utf8()
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY tanımlı değil. .env içine OPENAI_API_KEY=... ekleyin.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    scenario_system_prompt_only(client)
    scenario_few_shot(client)
    scenario_chain_of_thought(client)
    print()


if __name__ == "__main__":
    main()
