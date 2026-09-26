"""Source-shaped cached judgments remain well-defined through the audit pipeline."""
import copy
import hashlib
import io
import json

import pandas as pd

from judgecal.judge_audit import judge_accuracy_audit
from judgecal.llmbar import JUDGES, SUBSETS, extract_llmbar_labels, labels_csv_bytes


def source_documents():
    """Five source subsets, three caches, a repeated input, and both parse failures."""
    documents = {}
    for subset in SUBSETS:
        base = "Dataset/LLMBar/" + ("" if subset == "Natural" else "Adversarial/") + subset
        gold = []
        for i in range(12):
            instruction = 0 if subset == "Natural" and i == 1 else i
            gold.append({"input": f"{subset} instruction {instruction}",
                         "output_1": f"first {i}", "output_2": f"second {i}", "label": 1 + i % 2})
        documents[f"{base}/dataset.json"] = gold
        for judge in JUDGES:
            cached = []
            for i, item in enumerate(gold):
                # LLaMA2 deliberately makes consistent wrong choices: agreement != accuracy.
                choice = item["label"] if judge != "LLaMA2" else 3 - item["label"]
                forward = None if subset == "Natural" and judge == "GPT-4" and i == 2 else str(choice)
                reverse = None if subset == "Natural" and judge == "ChatGPT" and i == 3 else str(choice)
                result = lambda winner: {"completion": ["cached text", {}], "winner": winner}
                cached.append({**item, "results": [{"swap = False": result(forward),
                                                    "swap = True": result(reverse)}]})
            # Source order does not determine the gold/cache join.
            documents[f"{base}/evaluators/{judge}/Vanilla/result.json"] = cached[::-1]
    return documents


def sample_id(row):
    content = [row[key] for key in ("input", "output_1", "output_2")]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def test_extractor_csv_audit_preserves_canonical_choices_invalids_and_instruction_groups():
    source = source_documents()
    frame = extract_llmbar_labels(source)
    assert len(frame) == 60 * 3
    assert frame.sample_id.nunique() == 60
    assert frame.instruction_id.nunique() == 59
    assert (frame.forward_prediction == 0).sum() == 1
    assert (frame.reverse_prediction == 0).sum() == 1
    valid = frame.forward_prediction.ne(0) & frame.reverse_prediction.ne(0)
    assert frame.loc[valid, "forward_prediction"].equals(frame.loc[valid, "reverse_prediction"])
    serialized = labels_csv_bytes(frame)
    loaded = pd.read_csv(io.BytesIO(serialized), dtype={"sample_id": str, "instruction_id": str})
    pd.testing.assert_frame_equal(frame, loaded)
    before, summary = judge_accuracy_audit(frame, seeds=[6])
    after, loaded_summary = judge_accuracy_audit(loaded, seeds=[6])
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(summary, loaded_summary)
    # The always-wrong judge still agrees in both orders and is never rewarded by gold scoring.
    wrong = before.loc[before.judge == "LLaMA2"]
    assert wrong.heldout_accuracy_reference.eq(0).all()
    assert wrong.loc[wrong.method == "raw_proxy", "estimate"].eq(1).all()
    assert wrong.loc[wrong.method != "raw_proxy", "estimate"].eq(0).all()
    repeated = frame.drop_duplicates("sample_id").groupby("instruction_id").filter(lambda group: len(group) == 2)
    assert len(repeated) == 2
    for manifest in before.attrs["split_manifest"]:
        if manifest["cohort"] == "Adversarial":
            continue
        pools = [set(manifest[f"{pool}_sample_ids"]) for pool in ("labeled", "evaluation", "unused")]
        assert sum(set(repeated.sample_id) <= pool for pool in pools) == 1


def test_changed_hidden_source_gold_preserves_ids_and_all_estimation_inputs():
    source = source_documents()
    frame = extract_llmbar_labels(source)
    before, _ = judge_accuracy_audit(frame, seeds=[6], cohorts=["All"])
    hidden = set(before.attrs["split_manifest"][0]["evaluation_sample_ids"])
    changed = copy.deepcopy(source)
    # Change references consistently in source gold and all caches; predictions remain frozen.
    for rows in changed.values():
        for row in rows:
            if sample_id(row) in hidden:
                row["label"] = 3 - row["label"]
    modified = extract_llmbar_labels(changed)
    pd.testing.assert_frame_equal(frame.drop(columns="reference_label"), modified.drop(columns="reference_label"))
    after, _ = judge_accuracy_audit(modified, seeds=[6], cohorts=["All"])
    kept = ["estimate", "selected_power", "power_method", "interval_low", "interval_high",
            "standard_error", "interval_width", "split_sha256"]
    pd.testing.assert_frame_equal(before[kept], after[kept])
    assert before.attrs["split_manifest"] == after.attrs["split_manifest"]
    assert not before.heldout_accuracy_reference.equals(after.heldout_accuracy_reference)
