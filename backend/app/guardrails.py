from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.agents.middleware.types import Runtime, hook_config
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


PROMPT_INJECTION_BLOCKED_MESSAGE = (
    "I can't process that request because it appears to contain instructions to bypass "
    "or override the agent's safety controls."
)

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\b(ignore|disregard|forget|bypass|override)\b.{0,80}"
        r"\b(previous|prior|above|system|developer|original|all)\b.{0,80}"
        r"\b(instructions?|prompts?|rules?|policies|constraints?)\b",
        r"\b(reveal|print|show|display|dump|exfiltrate)\b.{0,80}"
        r"\b(system|developer)\s+(prompt|message|instructions?)\b",
        r"\b(system|developer)\s+(prompt|message|instructions?)\b.{0,80}"
        r"\b(reveal|print|show|display|dump|exfiltrate)\b",
        r"\bdo\s+not\s+(follow|obey)\b.{0,80}\b(system|developer|previous|above)\b",
        r"\b(jailbreak|DAN\s+mode)\b",
    )
)


class PromptInjectionMiddleware(AgentMiddleware):
    """Block obvious prompt-injection attempts before the next model call."""

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: dict[str, Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        text = _latest_untrusted_message_text(state.get("messages"))
        if text and _looks_like_prompt_injection(text):
            return {
                "jump_to": "end",
                "messages": [AIMessage(content=PROMPT_INJECTION_BLOCKED_MESSAGE)],
            }
        return None


def build_agent_guardrails(*, run_limit: int) -> list[AgentMiddleware]:
    return [
        PromptInjectionMiddleware(),
        ModelCallLimitMiddleware(run_limit=run_limit, exit_behavior="end"),
    ]


def _latest_untrusted_message_text(messages: Any) -> str:
    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return ""

    for message in reversed(messages):
        if _is_untrusted_message(message):
            return _message_text(message)
    return ""


def _is_untrusted_message(message: Any) -> bool:
    if isinstance(message, (HumanMessage, ToolMessage)):
        return True

    if isinstance(message, BaseMessage):
        return message.type in {"human", "tool"}

    if isinstance(message, dict):
        return message.get("role") in {"user", "tool"} or message.get("type") in {"human", "tool"}

    return False


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", message)
    if isinstance(content, str):
        return content

    if isinstance(content, Sequence):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return str(content)


def _looks_like_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PROMPT_INJECTION_PATTERNS)
