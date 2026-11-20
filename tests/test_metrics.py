"""Offline pieces of the evaluation: ranking metrics, tokenizer, BM25 and fusion."""

import json
from pathlib import Path

from langchain_core.documents import Document

from app.rag.ingest import chunk_ids, dedupe_chunks, strip_boilerplate
from app.rag.lexical import BM25Index, cosine_distance, reciprocal_rank_fusion, tokenize
from evals.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank


def test_ranking_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 5) == 0.5
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert reciprocal_rank(["a"], {"b"}) == 0.0
    assert abs(ndcg_at_k(["b"], {"b"}, 5) - 1.0) < 1e-9
    assert ndcg_at_k(["a"], {"b"}, 5) == 0.0
    assert 0 < ndcg_at_k(["a", "b"], {"b"}, 5) < 1
    assert mean([1.0, 0.0]) == 0.5 and mean([]) == 0.0


def test_tokenizer_folds_accents_and_case():
    assert tokenize("Antecipação de recebíveis, Pix!") == ["antecipacao", "de", "recebiveis", "pix"]
    assert tokenize("Wi-Fi 4G") == ["wi", "fi", "4g"]


def test_bm25_ranks_matching_documents_and_round_trips(tmp_path):
    index = BM25Index.from_texts(
        ["a", "b", "c"],
        ["antecipação de vendas e taxa", "pix recebimento conta", "maquininha get smart wi-fi"],
    )
    assert index.top("antecipacao", 3)[0][0] == "a"
    assert [doc_id for doc_id, _ in index.top("pix", 3)] == ["b"]
    assert index.top("nada", 3) == []
    path = tmp_path / "lexical.json"
    index.save(path)
    assert BM25Index.load(path).top("smart", 1)[0][0] == "c"


def test_reciprocal_rank_fusion_prefers_items_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert fused[0][0] == "b"
    assert {doc_id for doc_id, _ in fused} == {"a", "b", "c"}


def test_cosine_distance():
    assert abs(cosine_distance([1, 0], [1, 0])) < 1e-9
    assert abs(cosine_distance([1, 0], [0, 1]) - 1) < 1e-9
    assert cosine_distance([0, 0], [1, 0]) == 1.0


def test_scenarios_have_reference_answers_and_no_echoed_keywords():
    lines = (Path(__file__).resolve().parent.parent / "evals" / "scenarios.jsonl").read_text()
    scenarios = [json.loads(line) for line in lines.splitlines() if line.strip()]
    assert len(scenarios) >= 20
    for scenario in scenarios:
        assert scenario["reference_answer"], scenario["id"]
        question = scenario["message"].lower()
        for word in scenario.get("must_contain_any", []):
            assert word.lower() not in question, (scenario["id"], word)


def test_strip_boilerplate_keeps_the_product_text():
    page = (
        "## Já é cliente?\nNos informe CPF ou CNPJ\nVISITE\nO SITE\nOpen\n\n"
        "## Como podemos te ajudar?\nJá sou cliente\nAVANÇAR\n\n"
        "## crediário\nComo eu contrato o Crediário?\nPelo aplicativo.\n\n"
        "## Não é cliente?\nAdquira nossos produtos\n"
    )
    cleaned = strip_boilerplate(page)
    assert cleaned.startswith("## crediário")
    assert "Já é cliente" not in cleaned and "Não é cliente" not in cleaned
    assert "Pelo aplicativo." in cleaned


def test_dedupe_drops_hub_copies_but_keeps_the_owning_page():
    answer = (
        "Cada emissor de cartão estabelece o limite máximo de parcelas do crediário. "
        "O valor da venda é depositado na conta do estabelecimento em até dois dias úteis. "
        "A contratação é feita pelo aplicativo Getnet, na área de produtos digitais."
    )
    hub = Document(page_content=answer, metadata={"url": "https://site.getnet.com.br/duvidas/"})
    owner = Document(
        page_content=answer, metadata={"url": "https://site.getnet.com.br/duvidas/crediario/"}
    )
    product = Document(
        page_content=answer + " Conheça as condições especiais para a sua loja hoje mesmo.",
        metadata={"url": "https://site.getnet.com.br/crediario/"},
    )
    unique = Document(
        page_content="O Pix funciona 24 horas por dia, inclusive aos fins de semana e feriados. "
        "As chaves podem ser CPF, CNPJ, e-mail, telefone ou uma chave aleatória.",
        metadata={"url": "https://site.getnet.com.br/duvidas/pix/"},
    )
    short = Document(page_content="Taxas.", metadata={"url": "https://site.getnet.com.br/duvidas/"})
    kept = dedupe_chunks([hub, owner, product, unique, short])
    assert kept == [owner, product, unique, short]
    assert chunk_ids(kept)[:2] == [
        "site-getnet-com-br-duvidas-crediario-001",
        "site-getnet-com-br-crediario-001",
    ]
