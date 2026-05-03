from app.main import _source_card


def _chunk(score_breakdown: dict, **extra) -> dict:
    base = {
        "file": "fund.xlsx",
        "chunk_type": "row",
        "canonical_id": "hdfc top 100",
        "_score_breakdown": score_breakdown,
        "_score": 1.4,
    }
    base.update(extra)
    return base


def test_exact_id_match_renders_as_full_confidence():
    card = _source_card(_chunk({"exact_id": True, "text_sim": 0.6}))
    assert card["score"] == 1.0
    assert card["match"] == "exact"


def test_exact_cell_match_renders_as_high_confidence():
    card = _source_card(_chunk({"exact_id": False, "exact_cell": True, "text_sim": 0.4}))
    assert card["score"] == 0.95
    assert card["match"] == "field"


def test_semantic_match_uses_raw_cosine():
    card = _source_card(_chunk({"text_sim": 0.685}))
    assert card["score"] == 0.685
    assert card["match"] == "semantic"


def test_card_includes_canonical_id_and_type():
    card = _source_card(_chunk({"text_sim": 0.5}))
    assert card["canonical_id"] == "hdfc top 100"
    assert card["type"] == "row"


def test_rank_score_preserved_for_debugging():
    card = _source_card(_chunk({"exact_id": True, "text_sim": 0.6}))
    assert card["rank_score"] == 1.4
