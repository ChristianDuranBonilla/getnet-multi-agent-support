from langchain_core.messages import AIMessage

from app import repositories
from app.agents.router import router_node
from app.graph import after_dispatch, after_sanitize, after_specialist, build_graph
from app.rag import retriever as rag
from app.state import GuardrailResult, RouterDecision, initial_state
from tests.conftest import FakeRetriever


def test_low_confidence_forces_knowledge_and_handoff(use_structured):
    use_structured(
        RouterDecision(agent="support", intent="other", confidence=0.2, rationale="unclear")
    )
    update = router_node(initial_state("r1", "cliente1988", "hmm"))
    assert update["route"].agent == "knowledge"
    assert update["handoff"] is True


def test_edge_selectors():
    assert after_sanitize({"answer": "blocked"}) == "finalizer"
    assert after_sanitize({"answer": None}) == ["guardrail", "router"]
    decision = RouterDecision(agent="support", intent="device_issue", confidence=0.9)
    assert after_dispatch({"route": decision}) == "support"
    assert after_dispatch({"route": None}) == "knowledge"
    assert after_dispatch({"answer": "blocked", "route": decision}) == "finalizer"
    assert after_specialist({"handoff": True}) == "escalation"
    assert after_specialist({"handoff": False}) == "finalizer"


def test_graph_routes_product_question_to_knowledge(monkeypatch, use_structured_map, use_chat):
    use_structured_map(
        {
            GuardrailResult: GuardrailResult(allowed=True, category="ok", language="en"),
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
        initial_state("r1", "cliente1988", "Difference between Get Clássica and Get Smart?")
    )
    assert final["route"].agent == "knowledge"
    assert final["handoff"] is False
    assert final["sources"][0].type == "rag"
    nodes = [step.node for step in final["trace"]]
    assert set(nodes) == {"sanitize", "guardrail", "router", "knowledge", "finalizer"}
    assert nodes[0] == "sanitize" and nodes[-1] == "finalizer"


def test_graph_routes_deposit_question_to_support_with_tools(
    monkeypatch, use_structured_map, use_chat, repository
):
    use_structured_map(
        {
            GuardrailResult: GuardrailResult(allowed=True, category="ok", language="en"),
            RouterDecision: RouterDecision(
                agent="support", intent="account_or_transaction", confidence=0.9
            ),
        }
    )
    monkeypatch.setattr(repositories, "get_repository", lambda: repository)
    tool_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_settlement_status",
                "args": {"user_id": "cliente1988"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    use_chat(
        tool_turn,
        AIMessage(content="Your credit sale of R$ 450.00 will be deposited on 2026-12-05."),
    )
    final = build_graph().invoke(
        initial_state("r1", "cliente1988", "When will yesterday's sales be deposited?")
    )
    assert final["route"].agent == "support"
    assert [call.name for call in final["tool_calls"]] == ["get_settlement_status"]
    assert final["handoff"] is False
    assert "450" in final["answer"]


def test_unknown_user_escalates_with_ticket(monkeypatch, use_structured_map, use_chat, repository):
    use_structured_map(
        {
            GuardrailResult: GuardrailResult(allowed=True, category="ok", language="en"),
            RouterDecision: RouterDecision(
                agent="support", intent="account_or_transaction", confidence=0.9
            ),
        }
    )
    monkeypatch.setattr(repositories, "get_repository", lambda: repository)
    tool_turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_settlement_status",
                "args": {"user_id": "cliente0000"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )
    use_chat(
        tool_turn,
        AIMessage(content="I could not locate your account."),
        AIMessage(content="A human agent will follow up."),
    )
    final = build_graph().invoke(
        initial_state("r1", "cliente0000", "When will my sales be deposited?")
    )
    assert final["handoff"] is True
    assert final["ticket_id"] and final["ticket_id"].startswith("TCK-")
    assert final["ticket_id"] in final["answer"]
    assert "escalation" in [step.node for step in final["trace"]]
