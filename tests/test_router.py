from langchain_core.messages import AIMessage

from app.agents.router import router_node
from app.graph import after_dispatch, after_sanitize, build_graph
from app.rag import retriever as rag
from app.state import GuardrailResult, RouterDecision, initial_state
from tests.conftest import FakeRetriever, principal_for, tool_turn

OK = GuardrailResult(allowed=True, category="ok", language="en")


def test_low_confidence_forces_knowledge_and_handoff(use_structured):
    use_structured(
        RouterDecision(agent="support", intent="other", confidence=0.2, rationale="unclear")
    )
    update = router_node(initial_state("r1", principal_for("cliente1988"), "hmm"))
    assert update["route"].agent == "knowledge"
    assert update["plan"] == ["knowledge"]
    assert update["handoff"] is True


def test_router_receives_history_and_reports_tokens(use_structured):
    fake = use_structured(
        RouterDecision(agent="support", intent="device_issue", confidence=0.9, plan=["support"])
    )
    state = initial_state("r1", principal_for("cliente1988"), "It's the Smart.")
    state["summary"] = "The merchant reported an offline card machine."
    update = router_node(state)
    assert update["plan"] == ["support"]
    assert update["trace"][-1].tokens_in == 11
    assert "offline card machine" in fake.calls[0][0].content


def test_edge_selectors():
    assert after_sanitize({"answer": "blocked"}) == "finalizer"
    assert after_sanitize({"answer": None}) == ["guardrail", "router"]
    decision = RouterDecision(agent="support", intent="device_issue", confidence=0.9)
    assert after_dispatch({"route": decision}) == "support"
    assert after_dispatch({"route": None}) == "knowledge"
    assert after_dispatch({"answer": "blocked", "route": decision}) == "finalizer"


def test_graph_routes_product_question_to_knowledge(monkeypatch, use_structured_map, use_chat):
    use_structured_map(
        {
            OK.__class__: OK,
            RouterDecision: RouterDecision(
                agent="knowledge", intent="product_info", confidence=0.95
            ),
        }
    )
    use_chat(AIMessage(content="The Get Smart has a touchscreen; the Get Clássica does not."))
    hits = [
        (
            "Get Smart touchscreen",
            {"url": "https://site.getnet.com.br/todas-as-maquininhas/", "title": "Maquininhas"},
            0.2,
        )
    ]
    monkeypatch.setattr(rag, "get_retriever", lambda: FakeRetriever(hits))
    final = build_graph().invoke(
        initial_state(
            "r1", principal_for("cliente1988"), "Difference between Get Clássica and Get Smart?"
        )
    )
    assert final["route"].agent == "knowledge"
    assert final["current_agent"] == "knowledge"
    assert final["handoff"] is False
    assert final["sources"][0].type == "rag"
    assert final["evidence"][0].type == "rag" and final["evidence"][0].distance == 0.2
    nodes = [step.node for step in final["trace"]]
    assert nodes[:2] == ["recall", "sanitize"]
    assert {"guardrail", "router", "knowledge", "replan", "finalizer"} <= set(nodes)
    assert nodes[-1] == "remember"


def test_graph_routes_deposit_question_to_support_with_tools(
    use_structured_map, use_chat, use_repository
):
    use_structured_map(
        {
            OK.__class__: OK,
            RouterDecision: RouterDecision(
                agent="support", intent="account_or_transaction", confidence=0.9
            ),
        }
    )
    use_chat(
        tool_turn("get_settlement_status", {}),
        AIMessage(content="Your credit sale of R$ 450.00 will be deposited on 2026-12-05."),
    )
    final = build_graph().invoke(
        initial_state(
            "r1", principal_for("cliente1988"), "When will yesterday's sales be deposited?"
        )
    )
    assert final["route"].agent == "support"
    assert [call.name for call in final["tool_calls"]] == ["get_settlement_status"]
    assert final["handoff"] is False
    assert final["hops"] == 0
    assert "450" in final["answer"]
    assert final["evidence"][0].type == "tool"


def test_unknown_user_escalates_with_ticket(use_structured_map, use_chat, use_repository, tickets):
    use_structured_map(
        {
            OK.__class__: OK,
            RouterDecision: RouterDecision(
                agent="support", intent="account_or_transaction", confidence=0.9
            ),
        }
    )
    use_chat(
        tool_turn("get_settlement_status", {}),
        AIMessage(content="I could not locate your account."),
        AIMessage(content="A human agent will follow up."),
    )
    final = build_graph().invoke(
        initial_state(
            "r1", principal_for("cliente0000"), "When will my sales be deposited?", "cnv_test", 1
        )
    )
    assert final["handoff"] is True
    assert final["ticket_id"] and final["ticket_id"].startswith("TCK-")
    assert final["ticket_id"] in final["answer"]
    assert "ticket" in [step.node for step in final["trace"]]
    stored = tickets.get_ticket(final["ticket_id"])
    assert stored["merchant_id"] == "cliente0000" and stored["request_id"] == "r1"
