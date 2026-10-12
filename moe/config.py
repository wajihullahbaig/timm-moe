from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class Variant:
    """A model flavour: expert count, gate temperature and which auxiliary losses (with weights) to train with."""
    num_experts: int
    temperature: float
    loss_weights: Dict[str, float] = field(default_factory=dict)


VARIANTS = {
    # labels pre-assign classes to experts; the gate is supervised towards that assignment
    'assignment': Variant(10, 1.0, {'balance': 0.1, 'routing': 0.1, 'diversity': 0.1}),
    # unsupervised MoE: experts are pushed apart and kept evenly used
    'diversity': Variant(5, 2.0, {'balance': 0.1, 'diversity': 0.1}),
    # unsupervised MoE: classes are encouraged to stick to few experts
    'routed': Variant(5, 2.0, {'balance': 1.0, 'specialization': 1.0}),
}
