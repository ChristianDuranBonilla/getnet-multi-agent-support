"""Model factories: fallback model and model attribution."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import llm
from app.config import get_settings
from app.observability import model_of


def _raising(_messages):
    raise RuntimeError("504 DEADLINE_EXCEEDED")


def test_chat_falls_back_to_the_secondary_model(monkeypatch):
    built = []

    def fake_base_model(thinking, model_name):
        built.append(model_name)
        if model_name == get_settings().llm_model:
            return RunnableLambda(_raising)
        return RunnableLambda(
            lambda _m: AIMessage(
                content="from fallback", response_metadata={"model_name": model_name}
            )
        )

    monkeypatch.setattr(llm, "_base_model", fake_base_model)
    monkeypatch.setattr(get_settings(), "llm_fallback_model", "gemini-3.5-flash")
    response = llm.chat().invoke([])
    assert response.content == "from fallback"
    assert model_of(response) == "gemini-3.5-flash"
    assert built == [get_settings().llm_model, "gemini-3.5-flash"]


def test_chat_without_fallback_raises(monkeypatch):
    monkeypatch.setattr(llm, "_base_model", lambda thinking, model_name: RunnableLambda(_raising))
    monkeypatch.setattr(get_settings(), "llm_fallback_model", "")
    try:
        llm.chat().invoke([])
    except RuntimeError as exc:
        assert "504" in str(exc)
    else:
        raise AssertionError("expected the primary error to propagate")


def test_structured_model_reports_usage_and_model():
    raw = AIMessage(
        content="{}",
        usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        response_metadata={"model_name": "gemini-3.8-flash"},
    )
    wrapped = llm.StructuredModel(
        RunnableLambda(lambda _m: {"raw": raw, "parsed": {"ok": True}, "parsing_error": None})
    )
    assert wrapped.invoke([]) == {"ok": True}
    assert wrapped.last_usage == (5, 2)
    assert wrapped.last_model == "gemini-3.8-flash"
