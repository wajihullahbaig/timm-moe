"""Auxiliary MoE losses. Each takes (MoEOutput, labels) and returns a scalar; all are 0 at their optimum."""
import torch
import torch.nn.functional as F

EPS = 1e-8


def class_to_expert(num_classes, num_experts, device=None):
    """Expert index owning each class: contiguous, as balanced as possible."""
    assert num_experts <= num_classes, 'every expert needs at least one class'
    return torch.arange(num_classes, device=device) * num_experts // num_classes


def assignment_matrix(num_classes, num_experts):
    """(num_classes, num_experts) one-hot matrix of the class -> expert assignment."""
    return F.one_hot(class_to_expert(num_classes, num_experts), num_experts)


def balance_loss(out, labels):
    """0 when mean gate usage is uniform, num_experts - 1 when one expert takes everything."""
    usage = out.gate_weights.mean(0)
    return usage.numel() * usage.pow(2).sum() - 1


def routing_loss(out, labels):
    """Cross entropy pushing each sample's gate towards the expert assigned to its class."""
    num_experts = out.gate_weights.size(1)
    target = class_to_expert(out.logits.size(1), num_experts, labels.device)[labels]
    return F.nll_loss(out.gate_weights.clamp_min(EPS).log(), target)


def diversity_loss(out, labels):
    """Mean pairwise cosine similarity between different experts' outputs."""
    flat = F.normalize(out.expert_outputs.float().flatten(2), dim=2)
    similarity = torch.einsum('bed,bfd->ef', flat, flat) / flat.size(0)
    n = similarity.size(0)
    return (similarity.sum() - similarity.diagonal().sum()) / (n * (n - 1))


def specialization_loss(out, labels):
    """Mean entropy of each class's expert distribution: low when a class sticks to few experts."""
    onehot = F.one_hot(labels, out.logits.size(1)).to(out.gate_weights.dtype)
    counts = onehot.sum(0)
    present = counts > 0
    per_class = (onehot.t() @ out.gate_weights)[present] / counts[present, None]
    return -(per_class * per_class.clamp_min(EPS).log()).sum(1).mean()


LOSSES = {
    'balance': balance_loss,
    'routing': routing_loss,
    'diversity': diversity_loss,
    'specialization': specialization_loss,
}
