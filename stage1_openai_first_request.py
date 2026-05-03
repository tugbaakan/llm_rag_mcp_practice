"""
Aşama 1 — Görev 1: OpenAI API ile ilk chat completion isteği.
API anahtarı: proje kökündeki .env içinde OPENAI_API_KEY veya ortam değişkeni.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Script nereden çalıştırılırsa çalıştırılsın, proje kökündeki .env yüklensin
load_dotenv(Path(__file__).resolve().parent / ".env")


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "OPENAI_API_KEY tanımlı değil. "
            "Proje klasöründe .env dosyası oluşturup OPENAI_API_KEY=sk-... yazın "
            "(şablon: .env.example). Alternatif: $env:OPENAI_API_KEY='sk-...'",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Kısa ve net cevap ver."},
            {"role": "user", "content": "Merhaba! Tek cümleyle kendini tanıt."},
        ],
        max_tokens=80,
    )

    choice = response.choices[0]
    text = choice.message.content
    print(text or "(boş cevap)")


if __name__ == "__main__":
    main()
