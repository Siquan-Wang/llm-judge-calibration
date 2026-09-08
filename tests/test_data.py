"""Small invented labels exercise public-data parsing without network access."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from judgecal import data


def row(q=1, turn=1, a="alpha", b="beta", winner="model_a", judge="expert_0"):
    return dict(question_id=q, turn=turn, model_a=a, model_b=b, winner=winner, judge=judge)


def install_source(monkeypatch, human, gpt4):
    calls = []

    def load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        return {
            "human": SimpleNamespace(to_pandas=lambda: pd.DataFrame(human)),
            "gpt4_pair": SimpleNamespace(to_pandas=lambda: pd.DataFrame(gpt4)),
        }

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_canonical_winner_follows_model_identity():
    raw = pd.DataFrame([
        row(winner="model_a"), row(a="beta", b="alpha", winner="model_b"),
        row(winner="model_b"), row(a="beta", b="alpha", winner="model_a"),
        row(winner="tie"), row(a="beta", b="alpha", winner="tie (inconsistent)"),
    ])
    result = data._canonicalize(raw)
    assert result.label.tolist() == ["A", "A", "B", "B", "tie", "tie"]
    assert result.m_first.tolist() == ["alpha"] * 6
    assert result.input_reversed.tolist() == [False, True, False, True, False, True]
    assert result.inconsistent_order.tolist() == [False] * 5 + [True]
    assert raw.model_a.iloc[1] == "beta"


@pytest.mark.parametrize("bad", ["A", "model_c", "tie nonsense", "", None, 1])
def test_unknown_winner_rejected(bad):
    with pytest.raises(ValueError, match="winner must"):
        data._canonicalize(pd.DataFrame([row(winner=bad)]))


@pytest.mark.parametrize("a,b", [(None, "beta"), ("", "beta"), (" alpha", "beta"),
                                    ("alpha", "alpha"), (3, "beta")])
def test_invalid_model_identity_rejected(a, b):
    with pytest.raises(ValueError):
        data._canonicalize(pd.DataFrame([row(a=a, b=b)]))


@pytest.mark.parametrize("minimum", [0, -1, True, False, 1.0, "2", None])
def test_minimum_votes_is_positive_integer(minimum, monkeypatch):
    monkeypatch.setattr(data, "_load_source", lambda *_: pytest.fail("must validate before loading"))
    with pytest.raises(ValueError, match="positive integer"):
        data.load_mt_bench(minimum)


def test_join_deduplication_plurality_and_source_diagnostics(monkeypatch):
    human = [
        row(), row(a="beta", b="alpha", winner="model_b"),
        row(winner="model_b", judge="expert_1"),
        row(q=2, winner="tie"), row(q=2, winner="tie", judge="expert_1"),
        row(q=3),
    ]
    gpt4 = [row(a="beta", b="alpha", winner="tie (inconsistent)", judge="gpt4_pair"),
            row(q=2, winner="model_b", judge="gpt4_pair"), row(q=4, judge="gpt4_pair")]
    calls = install_source(monkeypatch, human, gpt4)
    result = data.load_mt_bench(min_human_votes=2)
    assert calls == [(data.MT_BENCH_DATASET, {"revision": data.MT_BENCH_REVISION, "token": False})]
    assert result.question_id.tolist() == [1, 2]
    assert result.human_winner.tolist() == ["tie", "tie"]
    assert result.n_human_votes.tolist() == [2, 2]
    assert result.gpt4_winner.tolist() == ["tie", "B"]
    assert result.gpt4_inconsistent_order.tolist() == [True, False]
    assert result.gpt4_input_reversed.tolist() == [True, False]
    assert result.gpt4_raw_winner.tolist() == ["tie (inconsistent)", "model_b"]
    assert result.loc[0, ["n_human_a", "n_human_b", "n_human_ties"]].tolist() == [1, 1, 0]
    assert result.attrs["dataset_revision"] == data.MT_BENCH_REVISION
    assert result.attrs["license"] == "CC-BY-4.0"
    counts = result.attrs["source_counts"]
    assert counts["human_duplicates_removed"] == 1
    assert counts["human_only_comparisons"] == counts["gpt4_only_comparisons"] == 1
    assert counts["aligned_inconsistent_order_after_filter"] == 1
    assert not result.attrs["response_alignment_checked"]
    install_source(monkeypatch, human[::-1], gpt4[::-1])
    pd.testing.assert_frame_equal(result, data.load_mt_bench(2))


def test_conflicting_annotator_votes_rejected(monkeypatch):
    install_source(monkeypatch, [row(), row(winner="model_b")], [row(judge="gpt4_pair")])
    with pytest.raises(ValueError, match="Conflicting human votes"):
        data.load_mt_bench()


def test_duplicate_combined_gpt4_judgment_rejected(monkeypatch):
    install_source(monkeypatch, [row()], [row(judge="gpt4_pair"), row(judge="gpt4_pair")])
    with pytest.raises(ValueError, match="one combined judgment"):
        data.load_mt_bench()


def test_filter_can_return_empty_with_schema_and_provenance(monkeypatch):
    install_source(monkeypatch, [row()], [row(judge="gpt4_pair")])
    result = data.load_mt_bench(2)
    assert result.empty
    assert "human_winner" in result and "question_id" in result
    assert result.attrs["source_counts"]["aligned_comparisons_after_filter"] == 0


def test_annotator_api_preserves_ids_and_distinct_votes(monkeypatch):
    install_source(monkeypatch, [row(q=91), row(q=91, a="beta", b="alpha", winner="model_b")], [])
    result = data.load_mt_bench_annotator_votes()
    assert result.columns[:6].tolist() == ["question_id", "turn", "model_a", "model_b", "judge", "label"]
    assert result.question_id.tolist() == [91]
    assert result.label.tolist() == ["A"]
    assert result.attrs["source_counts"]["human_duplicates_removed"] == 1


@pytest.mark.parametrize("column,bad", [("question_id", None), ("question_id", 1.5),
                                        ("turn", 0), ("judge", "")])
def test_required_prompt_and_annotator_identifiers(column, bad):
    record = row()
    record[column] = bad
    with pytest.raises(ValueError, match=column):
        data._prepare_split(pd.DataFrame([record]))


def test_response_alignment_is_checked_in_canonical_order(monkeypatch):
    a = [{"role": "assistant", "content": "Fixture response alpha"}]
    b = [{"role": "assistant", "content": "Fixture response beta"}]
    human = {**row(), "conversation_a": a, "conversation_b": b}
    judge = {**row(a="beta", b="alpha", winner="model_b", judge="gpt4_pair"),
             "conversation_a": b, "conversation_b": a}
    install_source(monkeypatch, [human], [judge])
    assert data.load_mt_bench().attrs["response_alignment_checked"]
    judge["conversation_b"] = [{"role": "assistant", "content": "Different fixture response"}]
    install_source(monkeypatch, [human], [judge])
    with pytest.raises(ValueError, match="different responses"):
        data.load_mt_bench()


def test_demo_split_keeps_every_turn_and_pair_of_prompt_together():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1] / "examples" / "demo_mtbench.py"
    spec = importlib.util.spec_from_file_location("demo_mtbench", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    df = pd.DataFrame([row(q=q, turn=t, b=b) for q in range(1, 9)
                       for t in (1, 2) for b in ("beta", "gamma")])
    audit, evaluation = demo.split_by_question(df, seed=0)
    assert set(audit.question_id).isdisjoint(evaluation.question_id)
    assert len(audit) + len(evaluation) == len(df)
    assert audit.question_id.nunique() == evaluation.question_id.nunique() == 4
    again, _ = demo.split_by_question(df.sample(frac=1, random_state=2), seed=0)
    assert set(audit.question_id) == set(again.question_id)
