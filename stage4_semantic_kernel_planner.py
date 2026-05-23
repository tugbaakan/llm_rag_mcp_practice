"""
Aşama 4 — Semantic Kernel ile özel skill (plugin) + SequentialPlanner.

SK 1.32+ sürümünde "skill" = kernel plugin (`@kernel_function` ile işaretli fonksiyonlar).
`SequentialPlanner` hedefe göre bir plan (XML) üretir ve skill fonksiyonlarını sırayla çalıştırır.

Not: `FunctionCallingStepwisePlanner` Groq'da llama-3.3-70b ile tool_use hatası verebilir;
bu örnek klasik SequentialPlanner kullanır (Groq ile uyumlu).

Önkoşul: `.env` içinde `GROQ_API_KEY`

Kullanım:
  python stage4_semantic_kernel_planner.py
  python stage4_semantic_kernel_planner.py --question "Özel soru"
  python stage4_semantic_kernel_planner.py -v
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import warnings
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.planners.sequential_planner import SequentialPlanner

load_dotenv(Path(__file__).resolve().parent / ".env")

SERVICE_ID = "groq"

DEFAULT_QUESTION = (
    "Bir okul kantininde 3 tepsi kurabiye var; her tepside 12 kurabiye. "
    "45 öğrenciye kişi başına en az 1 kurabiye dağıtılacak. "
    "Yeterli mi? Eksik veya fazla varsa sayıyı adım adım hesapla."
)


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


class SchoolSnackSkill:
    """Kantin / kurabiye dağıtımı için özel SK skill (plugin)."""

    @kernel_function(
        name="Multiply",
        description="İki tam sayıyı çarpar; tepsi × kurabiye gibi hesaplar için kullan.",
    )
    def multiply(
        self,
        a: Annotated[int, "Birinci çarpan"],
        b: Annotated[int, "İkinci çarpan"],
    ) -> Annotated[int, "Çarpım sonucu"]:
        return int(a) * int(b)

    @kernel_function(
        name="Subtract",
        description="İlk sayıdan ikinciyi çıkarır; fazla veya eksik kurabiye farkı için kullan.",
    )
    def subtract(
        self,
        minuend: Annotated[int, "Çıkarılacak sayı (büyük olan)"],
        subtrahend: Annotated[int, "Çıkan sayı"],
    ) -> Annotated[int, "Fark"]:
        return int(minuend) - int(subtrahend)

    @kernel_function(
        name="CompareCounts",
        description="Toplam kurabiye ile öğrenci sayısını karşılaştırır; yeterlilik özeti döndürür.",
    )
    def compare_counts(
        self,
        total_cookies: Annotated[int, "Toplam kurabiye sayısı"],
        students: Annotated[int, "Öğrenci sayısı"],
    ) -> Annotated[str, "Karşılaştırma özeti"]:
        total_cookies = int(total_cookies)
        students = int(students)
        if total_cookies >= students:
            surplus = total_cookies - students
            return f"Yeterli. {surplus} kurabiye fazla kalır."
        shortage = students - total_cookies
        return f"Yetersiz. {shortage} kurabiye eksik."


def _build_kernel() -> Kernel:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY tanımlı değil (.env).", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=SERVICE_ID,
            ai_model_id=model,
            async_client=AsyncOpenAI(
                api_key=key,
                base_url="https://api.groq.com/openai/v1",
            ),
        )
    )
    kernel.add_plugin(SchoolSnackSkill(), plugin_name="SchoolSnack")
    return kernel


def _step_label(step) -> str:
    plugin = getattr(step, "plugin_name", "") or ""
    name = getattr(step, "name", "") or ""
    if plugin and name:
        return f"{plugin}.{name}"
    fn = getattr(step, "function", None)
    if fn is not None:
        meta = getattr(fn, "metadata", None)
        if meta:
            return f"{meta.plugin_name}.{meta.name}"
    return name or "?"


def _print_plan(plan, *, verbose: bool) -> None:
    if not verbose:
        return
    print(f"\n--- Plan: {len(plan.steps)} adım ---", file=sys.stderr)
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {_step_label(step)}", file=sys.stderr)


async def run_planner(question: str, *, verbose: bool) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        kernel = _build_kernel()
        planner = SequentialPlanner(kernel, SERVICE_ID)
        plan = await planner.create_plan(question)
        _print_plan(plan, verbose=verbose)
        result = await plan.invoke(kernel)
    return str(result).strip()


def main() -> int:
    _ensure_utf8_stdio()
    p = argparse.ArgumentParser(description="Semantic Kernel: skill + SequentialPlanner")
    p.add_argument("--question", "-q", default=DEFAULT_QUESTION, help="Planner'ın çözeceği soru")
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Üretilen plan adımlarını stderr'e yaz",
    )
    args = p.parse_args()

    try:
        answer = asyncio.run(run_planner(args.question, verbose=args.verbose))
    except ImportError as e:
        print(
            f"Eksik paket: {e}\n"
            "Kurulum: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 1

    if answer:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
