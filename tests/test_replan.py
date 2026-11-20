"""Collaboration between specialists: planned hops, requested hops, caps and handoff."""

from langchain_core.messages import AIMessage

from app.agents.replan import after_replan, decide, next_step, replan_node
from app.config import get_settings
from app.graph import build_graph
from app.rag import retriever as rag
from app.state import (
    Evidence,
    GuardrailResult,
    PartialAnswer,
    RouterDecision,
    ToolCall,
    initial_state,
)
from tests.conftest import FakeRetriever, principal_for, tool_turn

OK = GuardrailResult(allowed=True, category="ok", language="en")
HITS = [
    (
        "Cada emissor de cartão estabelece o limite de parcelas.",
        {"url": "https://site.getnet.com.br/duvidas/crediario/", "title": "Crediário"},
        0.2,
    )
]


def test_next_step_rules_in_order():
    base = {"current_agent": "support", "hops": 0, "plan": ["support"]}
    assert next_step({**base, "handoff": True, "hop_request": {"agent": "knowledge"}}) == "ticket"
    assert (
        next_step({**base, "hop_request": {"agent": "knowledge", "question": "q"}}) == "knowledge"
    )
    assert next_step({**base, "plan": ["support", "knowledge"]}) == "knowledge"
    assert next_step({**base, "plan": ["support", "support"]}) == "finalizer"
    assert next_step({**base, "hop_request": {"agent": "support"}}) == "finalizer"
    assert next_step(base) == "finalizer"
    capped = {**base, "hops": 1, "plan": ["support", "knowledge"]}
    assert next_step({**capped, "hop_request": {"agent": "knowledge"}}) == "finalizer"


def test_replanner_owns_the_merchant_data_policy():
    """A failed lookup or an account under review escalates even if the model answered."""
    base = {"current_agent": "support", "hops": 0, "plan": ["support", "knowledge"]}
    healthy = ToolCall(name="get_user_profile", args={}, ok=True, account_status="active")
    review = ToolCall(name="get_user_profile", args={}, ok=True, account_status="under_review")
    failed = ToolCall(name="get_terminal_status", args={}, ok=False, summary="user_not_found")
    hop = ToolCall(name="request_product_knowledge", args={"question": "q"}, ok=False)
    assert next_step({**base, "tool_calls": [healthy]}) == "knowledge"
    assert next_step({**base, "tool_calls": [healthy, review]}) == "ticket"
    assert next_step({**base, "tool_calls": [failed]}) == "ticket"
    assert next_step({**base, "tool_calls": [hop]}) == "knowledge"
    sourced = {**base, "current_agent": "knowledge", "evidence": [Evidence(type="rag", ref="x")]}
    assert next_step({**sourced, "plan": ["knowledge"], "tool_calls": [review]}) == "finalizer"

    state = initial_state("r1", principal_for("cliente3050"), "Why are sales declined?")
    state.update(
        current_agent="support", answer="Your account is under review.", tool_calls=[review]
    )
    update = replan_node(state)
    assert update["next_node"] == "ticket" and update["handoff"] is True
    assert update["handoff_reason"] == "account under review"
    broken = replan_node({**state, "tool_calls": [failed]})
    assert broken["handoff_reason"] == "tool get_terminal_status failed: user_not_found"
    resolved = [{"reason": "account under review", "status": "resolved"}]
    assert decide({**base, "tool_calls": [review]}, resolved) == ("knowledge", "")
    still_open = [{"reason": "account under review", "status": "claimed"}]
    assert decide({**base, "tool_calls": [review]}, still_open) == (
        "ticket",
        "account under review",
    )
    other_cause = [
        {"reason": "tool get_terminal_status failed: user_not_found", "status": "resolved"}
    ]
    assert decide({**base, "tool_calls": [review]}, other_cause) == (
        "ticket",
        "account under review",
    )


def test_replan_node_moves_the_answer_aside_on_hop():
    state = initial_state("r1", principal_for("cliente1988"), "mixed question")
    state.update(
        current_agent="support",
        answer="R$ 450.00 pending.",
        tool_calls=[ToolCall(name="get_settlement_status", args={}, ok=True, summary="3 sales")],
        plan=["support", "knowledge"],
    )
    update = replan_node(state)
    assert update["next_node"] == "knowledge"
    assert update["hops"] == 1
    assert update["answer"] is None and update["sources"] == []
    partial = update["partial_answers"][0]
    assert isinstance(partial, PartialAnswer)
    assert partial.agent == "support" and partial.answer == "R$ 450.00 pending."
    assert partial.tool_summaries == ["get_settlement_status: 3 sales"]
    assert after_replan({**state, **update}) == "knowledge"
    finish = replan_node({**state, "plan": ["support"]})
    assert finish["next_node"] == "finalizer" and "hops" not in finish


def test_planned_hop_runs_both_specialists(
    monkeypatch, use_structured_map, use_chat, use_repository
):
    use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(
                agent="support",
                intent="account_or_transaction",
                confidence=0.9,
                plan=["support", "knowledge"],
            ),
        }
    )
    fake = use_chat(
        tool_turn("get_settlement_status", {}),
        AIMessage(content="Your credit sale of R$ 450.00 is scheduled for 2026-12-05."),
        AIMessage(content="R$ 450.00 on 2026-12-05; each issuer sets the instalment limit."),
    )
    monkeypatch.setattr(rag, "get_retriever", lambda: FakeRetriever(HITS))
    final = build_graph().invoke(
        initial_state(
            "r1",
            principal_for("cliente1988"),
            "When will I be paid for yesterday's sales and how many instalments are allowed?",
        )
    )
    nodes = [step.node for step in final["trace"]]
    assert nodes[nodes.index("support") :] == [
        "support",
        "replan",
        "knowledge",
        "replan",
        "finalizer",
        "remember",
    ]
    assert final["hops"] == 1 and final["plan"] == ["support", "knowledge"]
    assert final["current_agent"] == "knowledge"
    assert "instalment" in final["answer"]
    assert {source.type for source in final["sources"]} == {"tool", "rag"}
    assert {item.type for item in final["evidence"]} == {"tool", "rag"}
    knowledge_prompt = fake.calls[-1][0].content
    assert "R$ 450.00 is scheduled" in knowledge_prompt
    assert "get_settlement_status: 3 sales" in knowledge_prompt


def test_requested_hop_uses_the_question_from_support(
    monkeypatch, use_structured_map, use_chat, use_repository
):
    use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(agent="support", intent="device_issue", confidence=0.9),
        }
    )
    retriever = FakeRetriever(HITS)
    monkeypatch.setattr(rag, "get_retriever", lambda: retriever)
    use_chat(
        tool_turn("get_terminal_status", {}),
        tool_turn(
            "request_product_knowledge", {"question": "How do I reset a Get Clássica?"}, "call-2"
        ),
        AIMessage(content="Your Clássica is offline with error NET-02."),
        AIMessage(content="Offline with NET-02; to reset, re-enter the Wi-Fi password."),
    )
    final = build_graph().invoke(
        initial_state("r1", principal_for("cliente2001"), "My machine is offline, how do I reset?")
    )
    assert [call.name for call in final["tool_calls"]] == [
        "get_terminal_status",
        "request_product_knowledge",
    ]
    assert final["hops"] == 1
    assert retriever.queries[-1] == "How do I reset a Get Clássica?"
    assert final["hop_question"] == "How do I reset a Get Clássica?"
    assert "reset" in final["answer"]
    assert [source.ref for source in final["sources"] if source.type == "tool"] == [
        "get_terminal_status"
    ]


def test_hop_cap_is_enforced(monkeypatch, use_structured_map, use_chat, use_repository):
    monkeypatch.setattr(get_settings(), "max_hops", 0)
    use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(
                agent="support",
                intent="account_or_transaction",
                confidence=0.9,
                plan=["support", "knowledge"],
            ),
        }
    )
    use_chat(tool_turn("get_settlement_status", {}), AIMessage(content="R$ 450.00 pending."))
    final = build_graph().invoke(
        initial_state("r1", principal_for("cliente1988"), "mixed question")
    )
    assert final["hops"] == 0
    assert "knowledge" not in [step.node for step in final["trace"]]


def test_handoff_wins_over_hop(use_structured_map, use_chat, use_repository):
    use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(
                agent="support",
                intent="device_issue",
                confidence=0.9,
                plan=["support", "knowledge"],
            ),
        }
    )
    use_chat(
        tool_turn("get_user_profile", {}),
        AIMessage(content="Your account is under review."),
        AIMessage(content="unused"),
    )
    final = build_graph().invoke(
        initial_state("r1", principal_for("cliente3050"), "Why are my sales declined?")
    )
    nodes = [step.node for step in final["trace"]]
    assert "ticket" in nodes and "knowledge" not in nodes
    assert final["handoff"] is True and final["hops"] == 0


def test_unsourced_knowledge_answer_is_escalated():
    base = {"current_agent": "knowledge", "hops": 0, "plan": ["knowledge"], "evidence": []}
    assert next_step(base) == "ticket"
    update = replan_node({**base, "answer": "From memory."})
    assert update["next_node"] == "ticket" and update["handoff"] is True
    assert update["handoff_reason"] == "answer without sources"
    with_rag = {**base, "evidence": [Evidence(type="rag", ref="https://a", excerpt="x")]}
    assert next_step(with_rag) == "finalizer"
    with_web = {**base, "evidence": [Evidence(type="web", ref="google_search: q")]}
    assert next_step(with_web) == "finalizer"
    support_only = {**base, "current_agent": "support", "evidence": []}
    assert next_step(support_only) == "finalizer"
