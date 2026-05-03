"""
Aşama 1 — Görev 2: Groq API ile chat completion (ücretsiz API key: console.groq.com).

- Anahtar: .env içinde GROQ_API_KEY veya ortam değişkeni.
- Model: GROQ_MODEL ile tam ID verin; yoksa Llama (varsayılan) veya
  GROQ_BACKEND=mistral ile Mixtral kullanılır.

OpenAI scriptiyle aynı mesajlar kullanılıyor; çıktıda kısa bir karşılaştırma özeti var.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env")

# Groq konsolundaki güncel listeye göre değişebilir: https://console.groq.com/docs/models
MODEL_LLAMA = "llama-3.3-70b-versatile"
MODEL_MISTRAL = "mixtral-8x7b-32768"


def resolve_model() -> str:
    explicit = os.environ.get("GROQ_MODEL", "").strip()
    if explicit:
        return explicit
    backend = os.environ.get("GROQ_BACKEND", "llama").strip().lower()
    if backend in ("mistral", "mixtral"):
        return MODEL_MISTRAL
    return MODEL_LLAMA


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print(
            "GROQ_API_KEY tanımlı değil. https://console.groq.com üzerinden key alıp "
            ".env dosyasına GROQ_API_KEY=... ekleyin (.env.example şablonu).",
            file=sys.stderr,
        )
        sys.exit(1)

    model = resolve_model()
    client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": "Kısa ve net cevap ver."},
        {"role": "user", "content": "Merhaba! Tek cümleyle kendini tanıt."},
    ]

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=80,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    choice = response.choices[0]
    text = choice.message.content or "(boş cevap)"

    print(f"[model] {model}")
    print(f"[süre] {elapsed_ms:.0f} ms")
    print(text)
    print()
    print("--- OpenAI ile Groq: kısa karşılaştırma ---")
    print(
        "- İstemci: ikisi de `chat.completions.create` + mesaj listesi; "
        "Groq Python paketi OpenAI SDK’sına benzer bir yüzey sunar."
    )
    print(
        "- Uç nokta ve anahtar: OpenAI `api.openai.com` + OPENAI_API_KEY; "
        "Groq `api.groq.com` + GROQ_API_KEY (ayrı hesap)."
    )
    print(
        "- Hız / maliyet: Groq genelde çok düşük gecikme (LPU) ve cömert ücretsiz "
        "katman sunar; OpenAI tarafında model ve kullanıma göre ücretlendirme."
    )
    print(
        "- Model seçimi: Groq’ta Llama/Mixtral vb. sabit liste; OpenAI’da GPT ailesi. "
        "Üretim kalitesi göreve ve modele göre değişir — aynı promptla iki tarafta da dene."
    )


if __name__ == "__main__":
    main()
