# TEST-TAGS: modules=E,H; capabilities=multitask_learning,posterior; level=unit; cost=medium
import torch

from causal_schedule_lab.learning import (
    CausalCoreMultiTaskModel,
    GraphBatch,
    MultiTaskTargets,
    multitask_loss,
    pairwise_rank_pairs,
)
from causal_schedule_lab.trainers import train_cip_model


def test_multitask_model_trains_all_document_heads() -> None:
    batch = GraphBatch(
        node_features=torch.randn(6, 8),
        node_types=torch.tensor([0, 1, 2, 0, 1, 2]),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_types=torch.tensor([0, 1, 0, 2]),
        candidate_nodes=torch.tensor([0, 1, 2, 3]),
        candidate_batch=torch.tensor([0, 0, 1, 1]),
        graph_batch=torch.zeros(6, dtype=torch.long),
    )
    model = CausalCoreMultiTaskModel(
        numeric_dim=8,
        node_type_count=3,
        edge_type_count=3,
        risk_classes=4,
        hidden_dim=16,
    )
    output = model(batch)
    target = MultiTaskTargets(
        improvement=torch.tensor([2.0, 0.5]),
        validity=torch.tensor([1.0, 0.0]),
        log_cost=torch.tensor([0.0, 1.0]),
        risk=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
        ),
        closure_membership=torch.zeros(2, 6),
        path_membership=torch.zeros(2, 6),
        rank_pairs=pairwise_rank_pairs([2.0, 0.5]),
    )
    loss, parts = multitask_loss(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert {"improvement", "validity", "closure", "path", "ranking"} <= set(parts)
    history = train_cip_model(
        model,
        ((batch, target),),
        epochs=1,
        seed=0,
    )
    assert len(history) == 1
