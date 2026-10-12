import pytest
import torch
import torch.nn as nn

from moe.config import VARIANTS
from moe.data import augment, random_crop
from moe.engine import compute_losses
from moe.model import ExpertBank, MoEClassifier


@pytest.fixture(scope='module')
def model():
    return MoEClassifier(num_experts=5, expert_hidden_dim=16, pretrained=False)


def test_forward_shapes_and_gate_is_a_distribution(model):
    out = model(torch.randn(3, 3, 32, 32))
    assert out.logits.shape == (3, 10)
    assert out.gate_weights.shape == (3, 5)
    assert torch.allclose(out.gate_weights.sum(1), torch.ones(3), atol=1e-5)
    assert out.expert_outputs.shape[:2] == (3, 5)


def test_fused_expert_bank_experts_are_independent():
    bank = ExpertBank(dim=4, hidden_dim=6, num_experts=3).eval()
    x = torch.randn(2, 4, 5, 5)
    base = bank(x)
    # zeroing expert 1's second-conv weights must only change expert 1's output
    with torch.no_grad():
        bank.conv2.weight[4:8].zero_()  # output channels are laid out expert-major, dim=4 each
    changed = bank(x)
    assert torch.equal(base[:, 0], changed[:, 0]) and torch.equal(base[:, 2], changed[:, 2])
    assert not torch.equal(base[:, 1], changed[:, 1])


@pytest.mark.parametrize('name', list(VARIANTS))
def test_every_variant_gets_finite_losses_and_gradients(name):
    variant = VARIANTS[name]
    net = MoEClassifier(variant.num_experts, expert_hidden_dim=16, temperature=variant.temperature, pretrained=False)
    labels = torch.randint(0, 10, (8,))
    loss, terms = compute_losses(net(torch.randn(8, 3, 32, 32)), labels, variant.loss_weights)
    loss.backward()
    assert set(terms) == {'task', *variant.loss_weights}
    assert all(torch.isfinite(v) for v in terms.values())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_augmentation_keeps_shape_and_crop_can_shift():
    x = torch.randn(16, 3, 32, 32)
    assert augment(x).shape == x.shape
    cropped = random_crop(x)
    assert cropped.shape == x.shape and not torch.equal(cropped, x)


def test_evaluate_returns_accuracy_and_row_stochastic_usage(model):
    from moe.engine import evaluate

    class Loader:
        labels = torch.arange(10).repeat(2)
        batch_size = 20

        def __len__(self):
            return 1

        def __iter__(self):
            yield torch.randn(20, 3, 32, 32), self.labels

    acc, usage = evaluate(model, Loader(), amp=False)
    assert 0 <= acc <= 100
    assert usage.shape == (10, 5) and abs(usage.sum(1) - 1).max() < 1e-5


def test_early_stopping_flags_new_best_and_stops_after_patience():
    from moe.engine import EarlyStopping

    stopper = EarlyStopping(patience=2)
    assert stopper.update(50) and stopper.update(60)
    assert not stopper.update(59) and not stopper.should_stop
    assert not stopper.update(60) and stopper.should_stop
    assert stopper.update(61) and not stopper.should_stop
