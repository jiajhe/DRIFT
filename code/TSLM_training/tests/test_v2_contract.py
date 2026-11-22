import json
import numpy as np
import pytest
from v2_features import BASE, Inputs, WIDTHS
from prompt_templates_v2 import user_parts


@pytest.fixture(scope="module")
def sample():
    if not (BASE / "tasks/test.jsonl").exists():
        pytest.skip("Requires local history-v3")
    rows = [
        json.loads(line)
        for line in (BASE / "tasks/test.jsonl").read_text().splitlines()
    ]
    return Inputs(), rows


def test_interleaved_slots_and_future_independence(sample):
    source, rows = sample
    for t in range(4):
        row = next(r for r in rows if r["task"] == t)
        blocks = source.stock(row)
        assert list(map(len, blocks)) == ([10, 10] if t == 3 else [10] + [3] * 5)
        assert all(
            v.shape == (WIDTHS[key],) and np.isfinite(v).all()
            for block in blocks
            for key, p, v in block
        )
        mutated = dict(
            row, answer={}, future_ret=13, sender_cf_delta=[5] * 5, evidence="discard"
        )
        assert user_parts(row) == user_parts(mutated)
        for a, b in zip(blocks, source.stock(mutated)):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x[2], y[2])


def test_test_days_and_balanced_pair_winner(sample):
    _, rows = sample
    for t in range(4):
        xs = [r for r in rows if r["task"] == t]
        assert len(xs) == 1000 and len({r["date"] for r in xs}) == 500
    assert sum(r["answer"]["winner"] == "A" for r in rows if r["task"] == 3) == 500
    assert (
        sum(r["answer"]["confidence"] == "HIGH" for r in rows if r["task"] == 3) == 500
    )


def test_question_does_not_contain_soft_evidence(sample):
    transformers = pytest.importorskip("transformers")
    from v2_drift_tslm import MODEL, DriftTSLM
    from prompt_templates_v2 import legal_answer

    source, rows = sample
    reader = DriftTSLM.__new__(DriftTSLM)
    object.__setattr__(
        reader,
        "tokenizer",
        transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True),
    )
    for t in range(4):
        row = next(r for r in rows if r["task"] == t)
        headers, question = user_parts(row)
        blocks = source.stock(row)
        ids = reader.prefix(tuple(headers), question, tuple(map(len, blocks)))
        slots = [token for token in ids if token < 0]
        assert slots == list(range(-1, -1 - sum(map(len, blocks)), -1))
        assert legal_answer(row, row["answer"])
        assert not legal_answer(row, dict(row["answer"], delta=0))
        for j, header in enumerate(headers):
            num = sum(map(len, blocks[:j]))
            first = ids.index(-num - 1)
            preceding = reader.tokenizer.decode(
                [v for v in ids[max(0, first - 50) : first] if v >= 0]
            )
            assert preceding.endswith(header.splitlines()[-1])


def test_real_checkpoint_batch_masking():
    import os

    if os.environ.get("MARKETFLOW_GPU_TEST") != "1":
        pytest.skip("Requires a trained v2 checkpoint and GPU")

    import torch
    from api.main import model_at
    from v2_drift_tslm import WEIGHTS
    from v2_train import merged
    from prompt_templates_v2 import response

    path = WEIGHTS / "drift/best_valid.pt"
    model = model_at(path, path.stat().st_mtime_ns)
    valid = [
        json.loads(line)
        for line in (BASE / "tasks/valid.jsonl").read_text().splitlines()
    ]
    rows = [next(r for r in valid if r["task"] == t) for t in (0, 3)]
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        merged(model),
    ):
        _, mask, targets = model.prepare(rows, "real", None, True)
        assert (targets[mask == 0] == -100).all()
        for row, labels in zip(rows, targets, strict=True):
            assert labels[labels != -100].tolist() == model.encode(response(row)) + [
                model.eos
            ]
        mixed_loss, mixed_tokens = model.loss(rows)
        singles = [model.loss([r]) for r in rows]
        tokens = sum(n for _, n in singles)
        assert tokens == mixed_tokens
        solo_loss = sum(float(loss) for loss, _ in singles) / tokens
        assert abs(float(mixed_loss) / tokens - solo_loss) < 0.02
        # Check the inference boundary separately. Left padding must precede the real
        # prompt and the same first assistant token must be predicted in either batch.
        one_step = model.generate(rows, max_tokens=1)
        assert one_step == [model.generate([r], max_tokens=1)[0] for r in rows]


def test_separate_component_scales_and_local_observables(sample):
    from v2_features import slots, shares

    source, rows = sample
    row = rows[0]
    day, scaled = source.day(row["date"])
    native = slots(day)
    mean, scale = source.moments["component"]
    assert mean.shape == scale.shape == (3, 18)
    assert np.all(scale > 0) and np.ptp(scale[:, -2]) > 0
    np.testing.assert_allclose(
        scaled["component"], (native["component"] - mean) / scale
    )
    assert native["local"].shape[1] == 72
    np.testing.assert_array_equal(native["local"][:, 68], day["ens_score"])
    np.testing.assert_array_equal(native["local"][:, 69:], np.column_stack(shares(day)))


def test_answer_weight_starts_at_actual_json_token(sample):
    transformers = pytest.importorskip("transformers")
    from prompt_templates_v2 import response
    from v2_drift_tslm import MODEL, DriftTSLM

    reader = DriftTSLM.__new__(DriftTSLM)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    object.__setattr__(reader, "tokenizer", tok)
    object.__setattr__(reader, "eos", tok.convert_tokens_to_ids("<|im_end|>"))
    for t in range(4):
        row = next(r for r in sample[1] if r["task"] == t)
        txt = response(row)
        ans, tail = reader.answer_span(txt)
        assert ans == tok.encode(txt, add_special_tokens=False) + [reader.eos]
        boundary = txt.index("<answer>") + len("<answer>")
        offsets = tok(
            txt, add_special_tokens=False, return_offsets_mapping=True
        ).offset_mapping
        first = len(ans) - tail
        assert offsets[first - 1][1] <= boundary < offsets[first][1]
        assert reader.loss_mass(row) == first + 3 * tail


def test_pairwise_margin_uses_only_generated_evidence_and_counts_ties():
    from v2_pairwise import binary_metrics, winner_prefix, WINNER_PREFIX

    evidence = "<reasoning>The first stock has a broader flow signal. Its local ranking is higher.</reasoning>"
    generated = evidence + '\n<answer>{"winner":"A","confidence":"HIGH"}</answer>'
    prefix, valid = winner_prefix(generated)
    assert valid and prefix == evidence + "\n" + WINNER_PREFIX
    assert winner_prefix(generated.replace('"A"', '"B"')) == (prefix, True)
    fallback, valid = winner_prefix('<answer>{"winner":"B"}</answer>')
    assert not valid and fallback == "<reasoning></reasoning>\n" + WINNER_PREFIX
    result = binary_metrics([True, True, False, False], [1.0, 0.0, 0.0, -1.0])
    assert (result["correct"], result["ties"], result["win_rate"], result["auc"]) == (2, 2, 0.75, 0.875)
