"""Human in the loop: persistent tickets, interrupt on escalation, operator resume."""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import Command

from app import graph as graph_module
from app.main import app
from app.repositories.protocols import ConflictError
from app.state import GuardrailResult, RouterDecision, initial_state
from tests.conftest import DEMO_KEYS, OPERATOR_KEY, principal_for, tool_turn

OK = GuardrailResult(allowed=True, category="ok", language="en")
OPERATOR = {"X-API-Key": OPERATOR_KEY}


def escalate(graph, conversations, use_structured_map, use_chat, request_id="r1"):
    use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(agent="support", intent="device_issue", confidence=0.9),
        }
    )
    use_chat(
        tool_turn("get_user_profile", {}),
        AIMessage(content="Transactions are declined because your account is under review."),
    )
    conversation_id = conversations.create("cliente3050")
    thread_id = graph_module.new_thread_id(conversation_id, 1)
    state = initial_state(
        request_id,
        principal_for("cliente3050"),
        "My machine shows a decline error.",
        conversation_id,
        1,
        thread_id,
    )
    config = graph_module.thread_config(thread_id)
    result = graph.invoke(state, config=config)
    return conversation_id, config, result


def test_ticket_creation_is_idempotent_on_the_turn_thread(tickets):
    first = tickets.create_ticket("m", "s", "high", "cnv", "req-1", "cnv:1:aaaa")
    second = tickets.create_ticket("m", "s", "high", "cnv", "req-2", "cnv:1:aaaa")
    third = tickets.create_ticket("m", "s", "high", "cnv", "req-1", "cnv:2:bbbb")
    assert first == second != third
    foreign = tickets.create_ticket("other", "s", "high", "cnv2", "req-1", "cnv2:1:cccc")
    assert foreign not in (first, third)
    assert tickets.get_ticket(first, "other") is None
    assert tickets.get_ticket(first, "m")["status"] == "open"


def test_claim_and_reply_conflicts(tickets):
    ticket_id = tickets.create_ticket("m", "s", "high", "cnv", "req-1", "cnv:1")
    assert tickets.claim(ticket_id, "op-a")["claimed_by"] == "op-a"
    with pytest.raises(ConflictError) as excinfo:
        tickets.claim(ticket_id, "op-b")
    assert excinfo.value.code == "already_claimed"
    resolved = tickets.reply(ticket_id, "op-a", "Fixed.", "resolved")
    assert resolved["status"] == "resolved" and resolved["operator_reply"] == "Fixed."
    with pytest.raises(ConflictError) as excinfo:
        tickets.reply(ticket_id, "op-a", "Again.", "resolved")
    assert excinfo.value.code == "already_resolved"


def test_escalation_interrupts_and_operator_resumes(
    checkpointed_graph, conversations, tickets, use_structured_map, use_chat, use_repository
):
    conversation_id, config, result = escalate(
        checkpointed_graph, conversations, use_structured_map, use_chat
    )
    assert "__interrupt__" in result
    values = checkpointed_graph.get_state(config).values
    assert values["handoff"] is True
    ticket_id = values["ticket_id"]
    assert ticket_id in values["answer"]
    stored = tickets.get_ticket(ticket_id, "cliente3050")
    assert stored["status"] == "open" and stored["thread_id"] == config["configurable"]["thread_id"]
    turns = conversations.get(conversation_id, "cliente3050").turns
    assert [item.role for item in turns] == ["merchant", "assistant"]

    tickets.reply(ticket_id, "op-a", "The review is complete and sales are enabled.", "resolved")
    resumed = checkpointed_graph.invoke(
        Command(resume={"message": "The review is complete and sales are enabled."}), config=config
    )
    assert "__interrupt__" not in resumed
    assert resumed["operator_reply"] == "The review is complete and sales are enabled."
    assert resumed["answer"].startswith("Reply from our support team")
    turns = conversations.get(conversation_id, "cliente3050").turns
    assert [item.role for item in turns] == ["merchant", "assistant", "operator"]
    assert tickets.get_ticket(ticket_id)["status"] == "resolved"
    nodes = [step.node for step in resumed["trace"]]
    assert nodes[-2:] == ["await_operator", "remember"]


def test_graph_without_checkpointer_finishes_the_turn(
    conversations, tickets, use_structured_map, use_chat, use_repository
):
    conversation_id, _, result = escalate(
        graph_module.build_graph(), conversations, use_structured_map, use_chat
    )
    assert "__interrupt__" not in result
    assert result["ticket_id"] and tickets.get_ticket(result["ticket_id"])["status"] == "open"
    assert [item.role for item in conversations.get(conversation_id, "cliente3050").turns] == [
        "merchant",
        "assistant",
    ]


def test_operator_endpoints_round_trip(
    monkeypatch,
    checkpointed_graph,
    conversations,
    tickets,
    use_structured_map,
    use_chat,
    use_repository,
):
    monkeypatch.setattr(graph_module, "get_graph", lambda: checkpointed_graph)
    use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(agent="support", intent="device_issue", confidence=0.9),
        }
    )
    use_chat(
        tool_turn("get_user_profile", {}),
        AIMessage(content="Transactions are declined because your account is under review."),
    )
    merchant = {"X-API-Key": DEMO_KEYS["cliente3050"]}
    other = {"X-API-Key": DEMO_KEYS["cliente1988"]}
    with TestClient(app) as client:
        chat = client.post("/chat", json={"message": "Decline error."}, headers=merchant).json()
        assert chat["handoff"] is True and chat["ticket_id"]
        ticket_id = chat["ticket_id"]
        assert client.get(f"/tickets/{ticket_id}", headers=other).status_code == 404
        assert client.get(f"/tickets/{ticket_id}", headers=merchant).json()["status"] == "open"
        listed = client.get("/tickets?status=open", headers=OPERATOR).json()
        assert [item["ticket_id"] for item in listed] == [ticket_id]
        use_chat(
            tool_turn("get_user_profile", {}),
            AIMessage(content="Your account is still under review, so sales stay declined."),
        )
        repeat = client.post(
            "/chat",
            json={
                "message": "Still declined, any news?",
                "conversation_id": chat["conversation_id"],
            },
            headers=merchant,
        ).json()
        assert repeat["handoff"] is True and repeat["ticket_id"] == ticket_id
        assert "already open" in repeat["response"]
        assert [item["ticket_id"] for item in client.get("/tickets", headers=OPERATOR).json()] == [
            ticket_id
        ]
        assert client.post(f"/tickets/{ticket_id}/claim", headers=OPERATOR).status_code == 200
        second_operator = {"X-API-Key": DEMO_KEYS["operator_demo"]}
        assert client.get("/tickets?status=claimed", headers=second_operator).status_code == 200
        conflict = client.post(f"/tickets/{ticket_id}/claim", headers=second_operator)
        assert conflict.status_code == 409
        assert conflict.json()["detail"]["error"] == "already_claimed"
        stolen = client.post(
            f"/tickets/{ticket_id}/reply", json={"message": "Mine."}, headers=second_operator
        )
        assert stolen.status_code == 409
        assert stolen.json()["detail"]["error"] == "already_claimed"
        reply = client.post(
            f"/tickets/{ticket_id}/reply",
            json={"message": "Review complete, sales enabled.", "resolution": "resolved"},
            headers=OPERATOR,
        )
        assert reply.status_code == 200
        assert reply.json()["status"] == "resolved" and reply.json()["resumed"] is True
        again = client.post(f"/tickets/{ticket_id}/reply", json={"message": "x"}, headers=OPERATOR)
        assert again.status_code == 409
        assert again.json()["detail"]["error"] == "already_resolved"
        conversation = client.get(
            f"/conversations/{chat['conversation_id']}", headers=merchant
        ).json()
        roles = [item["role"] for item in conversation["turns"]]
        assert roles == ["merchant", "assistant", "merchant", "assistant", "operator"]
        assert "Review complete" in conversation["turns"][-1]["content"]
        assert client.get(f"/tickets/{ticket_id}", headers=merchant).json()["status"] == "resolved"

        # The data still says "under review", but the operator's decision stands: the
        # follow-up is answered from the reply and no new ticket is opened.
        use_chat(
            tool_turn("get_user_profile", {}),
            AIMessage(content="Our team confirmed the review is complete and sales are enabled."),
        )
        follow_up = client.post(
            "/chat",
            json={
                "message": "What did the support team say?",
                "conversation_id": chat["conversation_id"],
                "include_evidence": True,
            },
            headers=merchant,
        ).json()
        assert follow_up["handoff"] is False and follow_up["ticket_id"] is None
        assert "review is complete" in follow_up["response"]
        assert [item["type"] for item in follow_up["evidence"]][:1] == ["operator"]
        assert len(client.get("/tickets", headers=OPERATOR).json()) == 1


def test_interrupted_turn_resumes_after_a_restart(
    tmp_path, conversations, tickets, use_structured_map, use_chat, use_repository
):
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = tmp_path / "checkpoints.sqlite"
    first_connection = sqlite3.connect(path, check_same_thread=False)
    graph_before = graph_module.build_graph(
        SqliteSaver(first_connection, serde=graph_module.checkpoint_serializer())
    )
    conversation_id, config, result = escalate(
        graph_before, conversations, use_structured_map, use_chat
    )
    assert "__interrupt__" in result
    first_connection.close()

    second_connection = sqlite3.connect(path, check_same_thread=False)
    graph_after = graph_module.build_graph(
        SqliteSaver(second_connection, serde=graph_module.checkpoint_serializer())
    )
    resumed = graph_after.invoke(Command(resume={"message": "Review done."}), config=config)
    assert resumed["operator_reply"] == "Review done."
    roles = [item.role for item in conversations.get(conversation_id, "cliente3050").turns]
    assert roles == ["merchant", "assistant", "operator"]


def test_late_operator_reply_is_stored_without_a_waiting_thread(
    monkeypatch,
    checkpointed_graph,
    conversations,
    tickets,
    use_structured_map,
    use_chat,
    use_repository,
):
    monkeypatch.setattr(graph_module, "get_graph", lambda: checkpointed_graph)
    use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(agent="support", intent="device_issue", confidence=0.9),
        }
    )
    use_chat(
        tool_turn("get_user_profile", {}),
        AIMessage(content="Transactions are declined because your account is under review."),
    )
    merchant = {"X-API-Key": DEMO_KEYS["cliente3050"]}
    with TestClient(app) as client:
        chat = client.post("/chat", json={"message": "Decline error."}, headers=merchant).json()
        ticket_id = chat["ticket_id"]
        first = client.post(
            f"/tickets/{ticket_id}/reply",
            json={"message": "Can you send the terminal serial?", "resolution": "needs_merchant"},
            headers=OPERATOR,
        ).json()
        assert first["status"] == "waiting_merchant" and first["resumed"] is True
        second = client.post(
            f"/tickets/{ticket_id}/reply",
            json={"message": "Thanks, the review is complete.", "resolution": "resolved"},
            headers=OPERATOR,
        ).json()
        assert second["status"] == "resolved" and second["resumed"] is False
        conversation = client.get(
            f"/conversations/{chat['conversation_id']}", headers=merchant
        ).json()
    roles = [item["role"] for item in conversation["turns"]]
    assert roles == ["merchant", "assistant", "operator", "operator"]
    assert "review is complete" in conversation["turns"][-1]["content"]
