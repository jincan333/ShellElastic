import pytest
import torch

from elastic.models import DenseElasticConfig, DenseElasticLM, ShellConfig, ShellElasticLM
from elastic.train.losses import LinearCrossEntropy
from elastic.train.objective import Objective, ObjectiveConfig
from elastic.train.selection import WidthSelector, parse_step_rule


def shell(**overrides):
    cfg = dict(vocab_size=128, seq_len=16, n_layers=2, base_width=16, supported=[2, 3, 4], head_dim=16,
               ffn_hidden=16, proxy_width=32, fused_norm=False)
    cfg.update(overrides)
    torch.manual_seed(0)
    return ShellElasticLM(ShellConfig(**cfg)).double()


def batch(vocab=128, seq=16):
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, vocab, (2, seq), generator=gen)
    tgt = torch.roll(ids, -1, dims=1)
    tgt[0, :3] = -100
    return ids, tgt


def test_step_rules():
    assert parse_step_rule("1/5") == (1, 5)
    assert parse_step_rule("0") is None and parse_step_rule(0) is None
    sel = WidthSelector([6, 7, 8], "schedule", {6: (1, 3), 7: (2, 3), 8: (3, 3)})
    assert [sel(t) for t in (1, 2, 3, 4)] == [[6], [7], [8], [6]]


@pytest.mark.parametrize("strategy", ["random", "sandwich", "max_random"])
def test_stochastic_selection_is_a_function_of_the_step(strategy):
    a = WidthSelector([1, 2, 3, 4], strategy, seed=3)
    b = WidthSelector([1, 2, 3, 4], strategy, seed=3)
    picks = [tuple(a(t)) for t in range(1, 200)]
    assert picks == [tuple(b(t)) for t in range(1, 200)]
    if strategy == "sandwich":
        assert all(p[0] == 1 and p[-1] == 4 and len(p) == 3 for p in picks)
    if strategy == "max_random":
        assert all(p[-1] == 4 and len(p) == 2 for p in picks)
    if strategy == "random":
        assert set(p[0] for p in picks) == {1, 2, 3, 4}


def test_joint_loss_gradient_matches_weighted_sum():
    model = shell()
    ids, tgt = batch()
    weights = [1.0, 2.0, 4.0]
    objective = Objective(ObjectiveConfig(loss_weights=weights, z_loss=0.0), model)
    metrics = objective.train_step(model, ids, tgt, step=1)
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    outs = model(ids, targets=tgt, loss_fn=LinearCrossEntropy())
    manual = sum(w * outs[s].loss for s, w in zip([2, 3, 4], weights)) / sum(weights)
    manual.backward()
    assert metrics["loss"].item() == pytest.approx(manual.item())
    for n, p in model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(got[n], p.grad)


def test_dense_per_width_backward_matches_joint_loss():
    torch.manual_seed(0)
    model = DenseElasticLM(DenseElasticConfig(vocab_size=128, seq_len=16, n_layers=2, base_width=16,
                                              supported=[2, 3, 4], head_dim=16, ffn_hidden=32)).double()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("weight_delta"):
                p.normal_(0, 1e-3)
    ids, tgt = batch()
    objective = Objective(ObjectiveConfig(z_loss=0.0), model)
    objective.train_step(model, ids, tgt, step=1)
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    outs = model(ids, targets=tgt, loss_fn=LinearCrossEntropy())
    (sum(o.loss for o in outs.values()) / 3).backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(got[n], p.grad)


def test_unselected_widths_are_logged_without_gradient():
    model = shell()
    ids, tgt = batch()
    objective = Objective(ObjectiveConfig(selection="random", seed=0), model)
    step = next(t for t in range(1, 50) if objective.selector(t) == [2])
    metrics = objective.train_step(model, ids, tgt, step=step)
    assert {"loss_width_32", "loss_width_48", "loss_width_64"} <= set(metrics)
    assert model.blocks[0].attn.qkv.weight_2.grad is None  # the widest group was not trained
    assert model.readouts["2"].weight.grad is not None


@pytest.mark.parametrize("teacher", ["max", "ensemble", "oracle"])
def test_distillation_runs(teacher):
    model = shell()
    ids, tgt = batch()
    objective = Objective(ObjectiveConfig(distill=teacher, distill_weights=[0.5, 0.5, 0.5 if teacher != "max" else 0.0],
                                          ensemble_weighting="loss_weight"), model)
    metrics = objective.train_step(model, ids, tgt, step=1)
    assert "kd_width_32" in metrics and torch.isfinite(metrics["kd_width_32"])


def test_validation_reports_every_width():
    model = shell()
    ids, tgt = batch()
    metrics = Objective(ObjectiveConfig(), model).validation_step(model, ids, tgt)
    assert {"loss", "loss_width_32", "loss_width_48", "loss_width_64", "loss_width_avg"} <= set(metrics)
    assert all(p.grad is None for p in model.parameters())
