import numpy as np
import mujoco

from scripts.diagnose_g1_wbc_mjx_contact_force_oracle import (
    contact_api_floor_forces,
    load_rollout_array,
)


def test_contact_api_floor_forces_sums_mujoco_contact_force_normals() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.001" cone="pyramidal" solver="Newton"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="left" pos="0 0 0.04">
              <joint type="free"/>
              <geom name="left_foot_collision" type="box"
                    size="0.05 0.05 0.05" density="1000" condim="3"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "left_foot_collision",
    )

    forces = contact_api_floor_forces(
        model,
        data,
        floor_geom_ids=(floor_id,),
        left_foot_geom_ids=(left_id,),
        right_foot_geom_ids=(),
        other_robot_geom_ids=(),
    )

    assert forces[0] > 0.0
    assert forces[1] == 0.0
    assert forces[2] == 0.0


def test_load_rollout_array_squeezes_single_env_dimension(tmp_path) -> None:
    rollout = tmp_path / "rollout.npz"
    qpos = np.zeros((3, 1, 4), dtype=np.float32)
    qpos[:, 0, 3] = 1.0
    np.savez(rollout, qpos=qpos)

    loaded = load_rollout_array(rollout, "qpos")

    assert loaded.shape == (3, 4)
    np.testing.assert_allclose(loaded[:, 3], 1.0)
