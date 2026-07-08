import torch

from spider.tasks.g1_wbc.rollout import _contact_normal_force_from_solver_rows


def test_contact_normal_force_from_solver_rows_decodes_pyramidal_rows() -> None:
    efc_force = torch.tensor([[1.0, 2.0, 3.0, 4.0, 50.0, 6.0]])
    worldid = torch.tensor([0, 0], dtype=torch.long)
    address = torch.tensor(
        [
            [0, 1, 2, 3],
            [5, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    dim = torch.tensor([3, 1], dtype=torch.long)

    force = _contact_normal_force_from_solver_rows(
        efc_force,
        worldid,
        address,
        dim,
    )

    torch.testing.assert_close(force, torch.tensor([10.0, 6.0]))
