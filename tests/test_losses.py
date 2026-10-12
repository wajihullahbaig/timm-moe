import pytest
import torch

from moe.losses import (assignment_matrix, balance_loss, class_to_expert, diversity_loss, routing_loss,
                        specialization_loss)
from moe.model import MoEOutput


def make_output(gate_weights, expert_outputs=None, num_classes=10):
    batch, experts = gate_weights.shape
    if expert_outputs is None:
        expert_outputs = torch.rand(batch, experts, 4, 2, 2)
    return MoEOutput(torch.zeros(batch, num_classes), gate_weights, expert_outputs)


@pytest.mark.parametrize('experts,sizes', [(10, [1] * 10), (5, [2] * 5), (3, [4, 3, 3])])
def test_class_assignment_is_balanced_and_contiguous(experts, sizes):
    owner = class_to_expert(10, experts)
    assert torch.bincount(owner).tolist() == sizes
    assert (owner[1:] >= owner[:-1]).all()
    assert assignment_matrix(10, experts).sum(1).eq(1).all()


def test_balance_loss_zero_when_uniform_and_max_when_collapsed():
    uniform = make_output(torch.full((8, 5), 0.2))
    collapsed = make_output(torch.eye(5)[[0] * 8])
    assert balance_loss(uniform, None).item() == pytest.approx(0, abs=1e-6)
    assert balance_loss(collapsed, None).item() == pytest.approx(4)


def test_routing_loss_low_when_gate_matches_assignment():
    labels = torch.arange(10)
    perfect = make_output(torch.eye(10).clamp(min=1e-6))
    wrong = make_output(torch.eye(10)[(labels + 1) % 10].clamp(min=1e-6))
    assert routing_loss(perfect, labels).item() < 1e-4
    assert routing_loss(wrong, labels).item() > 5


def test_diversity_loss_is_one_for_identical_and_zero_for_orthogonal_experts():
    same = torch.rand(4, 1, 8).expand(4, 3, 8).reshape(4, 3, 2, 2, 2)
    orthogonal = torch.eye(3).repeat(4, 1, 1).reshape(4, 3, 3, 1, 1)
    gate = torch.full((4, 3), 1 / 3)
    assert diversity_loss(make_output(gate, same), None).item() == pytest.approx(1, abs=1e-5)
    assert diversity_loss(make_output(gate, orthogonal), None).item() == pytest.approx(0, abs=1e-5)


def test_specialization_loss_ignores_absent_classes_and_rewards_sharp_routing():
    labels = torch.tensor([0, 0, 1, 1])
    sharp = make_output(torch.eye(2)[[0, 0, 1, 1]])
    blurry = make_output(torch.full((4, 2), 0.5))
    assert specialization_loss(sharp, labels).item() == pytest.approx(0, abs=1e-5)
    assert specialization_loss(blurry, labels).item() == pytest.approx(0.6931, abs=1e-3)
