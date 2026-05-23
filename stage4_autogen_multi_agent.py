"""
Aşama 4 — AutoGen ile iki ajanlı (AssistantAgent + UserProxyAgent) görev çözümü.

AutoGen 0.7+ (autogen-agentchat) API kullanır; Python 3.13 ile uyumludur.
`RoundRobinGroupChat` içinde AssistantAgent çözer, UserProxyAgent (LLM destekli
inceleme) cevabı kontrol eder; gerekirse assistant düzeltir.

Önkoşul: `.env` içinde `GROQ_API_KEY`

Kullanım:
  python stage4_autogen_multi_agent.py
  python stage4_autogen_multi_agent.py --task "Özel görev metni"
  python stage4_autogen_multi_agent.py -v
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_TASK = (
    "Bir okul kantininde 3 tepsi kurabiye var; her tepside 12 kurabiye. "
    "45 öğrenciye kişi başına en az 1 kurabiye dağıtılacak. "
    "Yeterli mi? Eksik veya fazla varsa sayıyı adım adım hesapla."
)

ASSISTANT_SYSTEM = (
    "Sen dikkatli bir matematik asistanısın. Görevi adım adım çöz, ara hesapları açık yaz. "
    "İlk yanıtında ve düzeltme turunda TERMINATE yazma. "
    "user_proxy geri bildirimi 'ONAY:' ile başlıyorsa son yanıtında net özeti yazıp "
    "son satırda yalnızca TERMINATE yaz. "
    "'DUZELT:' ile başlıyorsa belirtilen hataları düzeltip yeniden sun; TERMINATE yazma."
)

REVIEWER_SYSTEM = (
    "Sen titiz bir matematik kontrolörüsün. Asistanın çözümünü adım adım incele. "
    "Hesaplama ve mantık doğruysa yanıtın tamamı 'ONAY:' ile başlasın; ardından "
    "kısa onay cümlesi yaz. "
    "Hata veya eksik adım varsa yanıtın tamamı 'DUZELT:' ile başlasın; "
    "neyin yanlış olduğunu ve ne yapması gerektiğini net yaz. "
    "Kendi yanıtında TERMINATE kelimesini kullanma."
)


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _build_groq_client():
    from autogen_core.models import ModelFamily
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        print("GROQ_API_KEY tanımlı değil (.env).", file=sys.stderr)
        sys.exit(1)
    model = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
    return OpenAIChatCompletionClient(
        model=model,
        base_url="https://api.groq.com/openai/v1",
        api_key=key,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        include_name_in_message=False,
    )


def _last_message_text(messages: Sequence[object], source: str) -> str:
    for message in reversed(messages):
        if getattr(message, "source", None) != source:
            continue
        content = getattr(message, "content", "")
        return content if isinstance(content, str) else str(content)
    return ""


async def _llm_review(model_client, task: str, assistant_answer: str, cancellation_token=None) -> str:
    from autogen_core.models import SystemMessage, UserMessage

    prompt = (
        f"Orijinal görev:\n{task}\n\n"
        f"Asistanın son cevabı:\n{assistant_answer}\n\n"
        "İncele ve ONAY: veya DUZELT: ile yanıt ver."
    )
    result = await model_client.create(
        [SystemMessage(content=REVIEWER_SYSTEM), UserMessage(content=prompt, source="user_proxy")],
        cancellation_token=cancellation_token,
    )
    content = result.content
    return content if isinstance(content, str) else str(content)


class ReviewingUserProxyAgent:
    """UserProxyAgent türevi: mesaj geçmişindeki asistan cevabını LLM ile inceler."""

    def __init__(
        self,
        name: str,
        *,
        model_client,
        task: str,
        description: str = "Çözümü inceleyen kullanıcı vekili.",
    ) -> None:
        from autogen_agentchat.agents import UserProxyAgent

        self._model_client = model_client
        self._task = task
        # RoundRobinGroupChat arayüzü için UserProxyAgent ile aynı yüzey
        self._delegate = UserProxyAgent(name=name, description=description)
        self.name = name
        self.description = description

    async def on_messages(self, messages, cancellation_token):
        async for message in self.on_messages_stream(messages, cancellation_token):
            if type(message).__name__ == "Response":
                return message
        raise AssertionError("ReviewingUserProxyAgent did not yield a Response.")

    async def on_messages_stream(self, messages, cancellation_token) -> AsyncGenerator[object, None]:
        from autogen_agentchat.base import Response
        from autogen_agentchat.messages import TextMessage

        assistant_answer = _last_message_text(messages, "assistant")
        if not assistant_answer:
            feedback = "Henüz çözüm yok; adım adım hesapla ve sun."
        else:
            feedback = await _llm_review(
                self._model_client,
                self._task,
                assistant_answer,
                cancellation_token,
            )

        yield Response(chat_message=TextMessage(content=feedback, source=self.name))

    async def on_reset(self, cancellation_token=None) -> None:
        await self._delegate.on_reset(cancellation_token)

    @property
    def produced_message_types(self):
        return self._delegate.produced_message_types


async def run_team(task: str, *, verbose: bool) -> str:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console

    model_client = _build_groq_client()

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=ASSISTANT_SYSTEM,
        description="Matematik görevlerini adım adım çözen asistan.",
    )
    user_proxy = ReviewingUserProxyAgent(
        name="user_proxy",
        model_client=model_client,
        task=task,
        description="Asistan cevabını LLM ile inceleyen kullanıcı vekili.",
    )

    termination = TextMentionTermination("TERMINATE", sources=["assistant"]) | MaxMessageTermination(
        max_messages=12
    )
    team = RoundRobinGroupChat(
        [assistant, user_proxy],
        termination_condition=termination,
        max_turns=8,
    )

    stream = team.run_stream(task=task)
    if verbose:
        await Console(stream)
        return ""

    last_text = ""
    async for event in stream:
        if type(event).__name__ == "TaskResult" and getattr(event, "messages", None):
            for msg in reversed(event.messages):
                if getattr(msg, "source", None) == "assistant" and getattr(msg, "content", None):
                    last_text = str(msg.content)
                    break
        elif getattr(event, "source", None) == "assistant" and getattr(event, "content", None):
            last_text = str(event.content)
    return last_text


def main() -> int:
    _ensure_utf8_stdio()
    p = argparse.ArgumentParser(description="AutoGen: AssistantAgent + UserProxyAgent")
    p.add_argument("--task", "-t", default=DEFAULT_TASK, help="İki ajanın çözeceği görev")
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Konuşma akışını (Console) yazdır",
    )
    args = p.parse_args()

    try:
        result = asyncio.run(run_team(args.task, verbose=args.verbose))
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

    if not args.verbose and result:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
