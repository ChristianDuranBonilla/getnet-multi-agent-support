"""Input guardrail: deterministic PII masking and injection detection, then an LLM check."""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm
from app.agents.prompts import load_prompt
from app.observability import timed_node
from app.state import AgentState, GuardrailResult

logger = logging.getLogger(__name__)
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
CPF_PATTERN = re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)")
INJECTION_PATTERNS = (
    re.compile(
        r"ignore (all |the |your )?(previous|prior|above) (instructions|rules|prompts?)", re.I
    ),
    re.compile(
        r"(reveal|print|show|repeat) (me )?(your |the )?(system|hidden|secret) "
        r"(prompt|instructions)",
        re.I,
    ),
    re.compile(r"\byou are now\b.*\b(dan|jailbroken|unrestricted)\b", re.I),
    re.compile(r"(ignora|ignore) (todas )?(las|as) instru", re.I),
)
SAFE_REPLIES = {
    "prompt_injection": {
        "en": "I can only help with questions about Getnet products, your account or your card "
        "machine.",
        "pt": "Só posso ajudar com dúvidas sobre produtos Getnet, sua conta ou sua maquininha.",
        "es": "Solo puedo ayudar con preguntas sobre productos Getnet, tu cuenta o tu terminal.",
    },
    "unsafe": {
        "en": "I can't help with that request. If you need support with your Getnet account, "
        "I'm here to help.",
        "pt": "Não posso ajudar com esse pedido. Se precisar de suporte com sua conta Getnet, "
        "estou à disposição.",
        "es": "No puedo ayudar con esa solicitud. Si necesitas soporte con tu cuenta Getnet, "
        "aquí estoy.",
    },
    "off_topic": {
        "en": "That's outside what I can help with. Ask me about Getnet products, deposits or "
        "your card machine.",
        "pt": "Isso está fora do que posso ajudar. Pergunte sobre produtos Getnet, depósitos ou "
        "sua maquininha.",
        "es": "Eso está fuera de lo que puedo ayudar. Pregúntame sobre productos Getnet, "
        "depósitos o tu terminal.",
    },
}


def luhn_valid(digits: str) -> bool:
    """Return whether a digit string passes the Luhn checksum.

    Args:
        digits: Digits only.

    Returns:
        True when the checksum is valid.
    """
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value = value * 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def cpf_valid(digits: str) -> bool:
    """Return whether an 11-digit string is a structurally valid Brazilian CPF.

    Args:
        digits: Eleven digits.

    Returns:
        True when both check digits match.
    """
    if len(digits) != 11 or len(set(digits)) == 1:
        return False
    for length in (9, 10):
        total = sum(int(digits[i]) * (length + 1 - i) for i in range(length))
        check = (total * 10) % 11 % 10
        if check != int(digits[length]):
            return False
    return True


def mask_pii(message: str) -> tuple[str, bool]:
    """Mask card numbers and CPFs in place.

    Args:
        message: Raw user message.

    Returns:
        A tuple of (masked_message, found_pii).
    """
    found = False

    def mask_card(match: re.Match) -> str:
        nonlocal found
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            found = True
            return f"**** **** **** {digits[-4:]}"
        return match.group(0)

    def mask_cpf(match: re.Match) -> str:
        nonlocal found
        digits = re.sub(r"\D", "", match.group(0))
        if cpf_valid(digits):
            found = True
            return "***.***.***-**"
        return match.group(0)

    masked = CARD_PATTERN.sub(mask_card, message)
    masked = CPF_PATTERN.sub(mask_cpf, masked)
    return masked, found


def looks_like_injection(message: str) -> bool:
    """Return whether the message matches a known prompt-injection pattern."""
    return any(pattern.search(message) for pattern in INJECTION_PATTERNS)


def safe_reply(category: str, language: str) -> str:
    """Return the canned reply for a blocked or off-topic category in the user's language."""
    replies = SAFE_REPLIES[category]
    return replies.get(language, replies["en"])


def classify(message: str) -> GuardrailResult:
    """Run the LLM classification step.

    Args:
        message: Message with PII already masked.

    Returns:
        The model's verdict, or a permissive default when the model fails.
    """
    try:
        result = llm.structured(GuardrailResult).invoke(
            [SystemMessage(content=load_prompt("guardrail")), HumanMessage(content=message)]
        )
    except Exception as exc:
        logger.warning("%s model call failed: %s: %s", "guardrail", type(exc).__name__, exc)
        return GuardrailResult(
            allowed=True, category="ok", language="en", reason="classifier unavailable"
        )
    if isinstance(result, dict):
        result = GuardrailResult.model_validate(result)
    return result


@timed_node("sanitize")
def sanitize_node(state: AgentState) -> dict:
    """Mask PII and block prompt injections deterministically, before any model call.

    Runs first so that the Guardrail classifier and the Router can execute in
    parallel on the already-masked message.

    Args:
        state: Current graph state.

    Returns:
        State update with the masked `message`, `pii_detected` and, for injections,
        the `guardrail` verdict and a safe `answer`.
    """
    masked, found_pii = mask_pii(state["message"])
    update = {
        "message": masked,
        "pii_detected": found_pii,
        "trace_detail": "pii" if found_pii else "",
    }
    if looks_like_injection(masked):
        verdict = GuardrailResult(
            allowed=False, category="prompt_injection", language="en", reason="pattern match"
        )
        update.update(
            guardrail=verdict,
            language="en",
            answer=safe_reply("prompt_injection", "en"),
            trace_detail=verdict.category,
        )
    return update


@timed_node("guardrail")
def guardrail_node(state: AgentState) -> dict:
    """Classify the sanitized message and prepare a safe reply when it must not proceed.

    Args:
        state: Current graph state, already processed by `sanitize_node`.

    Returns:
        State update with `guardrail`, `language` and possibly an `answer`.
    """
    verdict = classify(state["message"])
    language = (verdict.language or "en").lower()[:2]
    if state.get("pii_detected") and verdict.category == "ok":
        verdict = verdict.model_copy(update={"category": "pii", "allowed": True})
    if verdict.category in ("unsafe", "prompt_injection"):
        verdict = verdict.model_copy(update={"allowed": False})
    update = {"guardrail": verdict, "language": language, "trace_detail": verdict.category}
    if verdict.category in ("unsafe", "prompt_injection", "off_topic"):
        update["answer"] = safe_reply(verdict.category, language)
    return update
