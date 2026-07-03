import tempfile
import unittest
from pathlib import Path

import torch

from spider.tasks.g1_wbc.constants import ACTION_DIM
from spider.tasks.g1_wbc.policy import load_wbc_actor


class WbcPolicyLoaderTest(unittest.TestCase):
    def test_sparse_transformer_checkpoint_fails_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "model.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "std": torch.ones(ACTION_DIM),
                        "actor.transformer_blocks.0.self_attention.in_proj_weight": (
                            torch.zeros(1, 1)
                        ),
                        "actor_task_embedder.task_projection.weight": torch.zeros(
                            1, 267
                        ),
                    }
                },
                checkpoint,
            )

            with self.assertRaisesRegex(
                ValueError,
                "SparseTrack transformer checkpoint.*separate policy/observation adapter",
            ):
                load_wbc_actor(checkpoint, device="cpu")

    def test_non_dict_checkpoint_fails_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "model.pt"
            torch.save([torch.zeros(1)], checkpoint)

            with self.assertRaisesRegex(
                ValueError,
                "Unsupported G1 WBC checkpoint format.*not a dict",
            ):
                load_wbc_actor(checkpoint, device="cpu")


if __name__ == "__main__":
    unittest.main()
