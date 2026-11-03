"""Pure config normalizers — no Hermes, no network."""

import types
from pathlib import Path
from unittest.mock import patch

from hindsight_hermes.settings import (
    _clears_min_scores,
    _normalize_min_scores,
    _normalize_observation_scopes,
    _normalize_string_list,
    _normalize_workspace_bank_routes,
    _parse_int_setting,
    _resolve_bank_id_template,
    _workspace_bank_id,
)


def test_retain_tags_accepts_csv_json_and_lists():
    assert _normalize_string_list("a, b ,a") == ["a", "b"]
    assert _normalize_string_list('["a", "b"]') == ["a", "b"]
    assert _normalize_string_list(["a", "a", "b"]) == ["a", "b"]
    assert _normalize_string_list(None) == []


def test_workspace_bank_routes_accepts_mapping_and_json(tmp_path: Path):
    root = tmp_path / "work"
    routes = _normalize_workspace_bank_routes({str(root): "work-bank"})
    assert routes == [(root.resolve(), "work-bank")]
    assert _normalize_workspace_bank_routes(f'{{"{root}": "work-bank"}}') == routes


def test_workspace_bank_routes_rejects_invalid_entries(tmp_path: Path):
    assert _normalize_workspace_bank_routes("not-json") == []
    assert _normalize_workspace_bank_routes(["not", "a", "mapping"]) == []
    assert _normalize_workspace_bank_routes({"relative/path": "bank"}) == []
    assert _normalize_workspace_bank_routes({str(tmp_path): ""}) == []


def test_workspace_bank_routes_ignores_unresolvable_paths(tmp_path: Path):
    with patch.object(Path, "resolve", side_effect=RuntimeError("symlink loop")):
        assert _normalize_workspace_bank_routes({str(tmp_path): "bank"}) == []


def test_workspace_bank_route_uses_nearest_ancestor(tmp_path: Path):
    root = tmp_path / "work"
    nested = root / "project"
    cwd = nested / "src"
    cwd.mkdir(parents=True)
    routes = {str(root): "parent-bank", str(nested): "project-bank"}
    assert _workspace_bank_id(str(cwd), routes) == "project-bank"


def test_workspace_bank_route_matches_path_components_not_prefixes(tmp_path: Path):
    route = tmp_path / "work"
    sibling = tmp_path / "worktree"
    sibling.mkdir()
    assert _workspace_bank_id(str(sibling), {str(route): "wrong-bank"}) is None


def test_parse_int_setting_falls_back_on_garbage():
    assert _parse_int_setting("30", 120) == 30
    assert _parse_int_setting("", 120) == 120
    assert _parse_int_setting("nope", 120) == 120
    assert _parse_int_setting(0, 120) == 0


def test_bank_id_template_sanitizes_and_collapses_empty_placeholders():
    assert _resolve_bank_id_template("hermes-{profile}", "hermes", profile="My Bot") == "hermes-My-Bot"
    assert _resolve_bank_id_template("hermes-{user}", "hermes", user="") == "hermes"
    assert _resolve_bank_id_template("", "hermes", user="x") == "hermes"
    # An unknown placeholder must not blow up the session — fall back to the static bank.
    assert _resolve_bank_id_template("hermes-{nope}", "hermes") == "hermes"


def test_observation_scopes_normalization():
    assert _normalize_observation_scopes("per_tag") == "per_tag"
    assert _normalize_observation_scopes("shared") == "shared"  # #4845: the server supports it
    assert _normalize_observation_scopes(["a", "b"]) == [["a", "b"]]
    assert _normalize_observation_scopes([["a"], ["b"]]) == [["a"], ["b"]]
    assert _normalize_observation_scopes("garbage") is None


def test_min_scores_accepts_a_mapping_or_a_json_object():
    assert _normalize_min_scores({"reranker": 0.25}) == {"reranker": 0.25}
    assert _normalize_min_scores('{"reranker": 0.25, "semantic": 1}') == {"reranker": 0.25, "semantic": 1.0}
    assert _normalize_min_scores(None) is None
    assert _normalize_min_scores("") is None
    assert _normalize_min_scores({}) is None


def test_min_scores_drops_invalid_floors_instead_of_sending_them():
    assert _normalize_min_scores("not json") is None
    assert _normalize_min_scores("[0.25]") is None  # a JSON value, but not an object
    assert _normalize_min_scores(0.25) is None
    # One bad entry must not discard the good one beside it.
    assert _normalize_min_scores({"reranker": "high", "semantic": 0.6, "keyword": True}) == {"semantic": 0.6}
    assert _normalize_min_scores({"reranker": float("nan")}) is None


def _result(**scores):
    return types.SimpleNamespace(scores=types.SimpleNamespace(**scores))


def test_clears_min_scores_is_inclusive_and_checks_every_floor():
    floors = {"semantic": 0.5, "reranker": 0.1}
    assert _clears_min_scores(_result(semantic=0.5, reranker=0.1), floors)
    assert not _clears_min_scores(_result(semantic=0.49, reranker=0.9), floors)
    assert not _clears_min_scores(_result(semantic=0.9, reranker=0.09), floors)


def test_clears_min_scores_rejects_a_result_that_does_not_report_the_floored_stage():
    # Surfaced by another retrieval arm: no semantic score, so it cannot clear a semantic floor.
    assert not _clears_min_scores(_result(semantic=None, reranker=0.5), {"semantic": 0.5})
    assert not _clears_min_scores(_result(reranker=0.5), {"semantic": 0.5})
    # ...but it is judged on the stages it does report.
    assert _clears_min_scores(_result(semantic=None, reranker=0.5), {"reranker": 0.1})


def test_clears_min_scores_keeps_a_result_with_no_scores_at_all():
    assert _clears_min_scores(types.SimpleNamespace(scores=None), {"semantic": 0.9})
    assert _clears_min_scores(types.SimpleNamespace(), {"semantic": 0.9})
