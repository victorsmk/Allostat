import numpy as np
import torch

from stressnet import federated as fed
from stressnet.models import MultiModalNet

MODS = ["wrist_bvp", "wrist_eda"]


def _windows(n, rng, labels=(0, 1, 2)):
    ws = []
    for i in range(n):
        y = int(labels[i % len(labels)])
        ws.append({"model_label": y, "subject": "x",
                   "wrist": {"BVP": (rng.normal(size=(3840, 1)) + 0.0 * y).astype(np.float32),
                             "EDA": (rng.normal(size=(240, 1)) + 3.0 * y).astype(np.float32)},
                   "chest": {}})
    return ws


def test_average_states_is_weighted_and_keeps_int_buffers():
    a = {"w": torch.tensor([0.0, 0.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([4.0, 8.0]), "n": torch.tensor(9)}
    out = fed.average_states([a, b], [1, 3])
    assert torch.allclose(out["w"], torch.tensor([3.0, 6.0]))
    assert int(out["n"]) == 5


def test_local_update_changes_weights_and_does_not_touch_the_global_state():
    rng = np.random.default_rng(0)
    model = MultiModalNet(MODS, num_classes=3)
    g = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    snapshot = {k: v.clone() for k, v in g.items()}
    new = fed.local_update(model, g, _windows(24, rng), 3, 1, 8, 1e-2, 0.0, torch.device("cpu"), seed=0)
    assert any(not torch.equal(new[k], snapshot[k]) for k in new if new[k].is_floating_point())
    assert all(torch.equal(g[k], snapshot[k]) for k in g)


def test_fedavg_learns_a_separable_problem_and_is_deterministic():
    rng = np.random.default_rng(1)
    clients = {f"c{i}": _windows(24, rng) for i in range(5)}
    test = _windows(60, np.random.default_rng(99))

    def evaluate(model):
        from stressnet.data.torch_dataset import WindowDataset
        from torch.utils.data import DataLoader
        pred, true = [], []
        with torch.no_grad():
            for b in DataLoader(WindowDataset(test), batch_size=30):
                pred.append(model({"chest": {}, "wrist": b["wrist"]}).argmax(1))
                true.append(b["label"])
        return {"acc": float((torch.cat(pred) == torch.cat(true)).float().mean())}

    runs = []
    for _ in range(2):
        model, hist = fed.fedavg(clients, MODS, rounds=8, local_epochs=2, batch_size=8, lr=3e-3, seed=0,
                                 eval_fn=evaluate, eval_every=4)
        runs.append((model, hist))
    assert runs[0][1][-1]["acc"] > 0.6                    # chance is 0.33
    assert [h["round"] for h in runs[0][1]] == [4, 8]
    for k, v in runs[0][0].state_dict().items():           # same seed -> identical global model
        assert torch.allclose(v.float(), runs[1][0].state_dict()[k].float())


def test_communication_cost_counts_both_directions():
    c = fed.communication_per_round(MultiModalNet(MODS, num_classes=3), n_clients=10)
    assert c["bytes_per_round"] == 10 * c["bytes_per_client"] == 10 * 2 * 4 * c["parameters"]
