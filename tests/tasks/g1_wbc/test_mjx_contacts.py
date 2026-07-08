import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.mjx_contacts import CONTACT_PROFILES, get_contact_profile


HGPT_4010_ASSET_DIR = Path(
    "/data_team/junsong/tracking/Humanoid-GPT/storage/assets/unitree_g1_4010"
)
HGPT_4010_TRACK_SCENE = HGPT_4010_ASSET_DIR / "scene_mjx_track.xml"
HGPT_4010_LOCO_SCENE = HGPT_4010_ASSET_DIR / "scene_mjx_loco.xml"


class MjxContactProfilesTest(unittest.TestCase):
    def test_contact_profiles_include_wxy_and_hgpt_references(self) -> None:
        self.assertEqual(
            set(CONTACT_PROFILES),
            {
                "wxy_parity",
                "wxy_explicit_pairs_7caps",
                "wxy_explicit_floor_pairs_7caps",
                "wxy_explicit_floor_leg_pairs_7caps",
                "wxy_explicit_floor_leg_foot147_pairs_7caps",
                "wxy_explicit_floor_leg_pair_param_parity_7caps",
                "hgpt_track_reference",
                "hgpt_loco_reference",
            },
        )

    def test_wxy_parity_profile_names_floor_and_feet(self) -> None:
        profile = get_contact_profile("wxy_parity")

        self.assertEqual(profile.name, "wxy_parity")
        self.assertIn("terrain", profile.floor_geom_names)
        self.assertNotIn("floor", profile.floor_geom_names)
        self.assertGreaterEqual(profile.max_contact_points, 128)
        self.assertGreaterEqual(profile.max_geom_pairs, 256)
        self.assertTrue(any("left" in name for name in profile.foot_body_names))
        self.assertTrue(any("right" in name for name in profile.foot_body_names))
        self.assertEqual(len(profile.foot_collision_geom_names), 14)
        self.assertEqual(profile.explicit_pair_names, ())
        self.assertTrue(profile.eligible_for_parity)

    def test_wxy_explicit_floor_leg_profile_prunes_upper_body_self_pairs(self) -> None:
        profile = get_contact_profile("wxy_explicit_floor_leg_pairs_7caps")

        self.assertTrue(profile.eligible_for_parity)
        self.assertEqual(profile.floor_geom_names, ("terrain",))
        self.assertEqual(profile.max_geom_pairs, 128)
        self.assertIn(
            ("robot/left_foot1_collision", "terrain"),
            profile.explicit_pair_names,
        )
        self.assertIn(
            ("robot/right_shin_collision", "robot/left_thigh_collision"),
            profile.explicit_pair_names,
        )
        self.assertNotIn(
            ("robot/torso_collision", "robot/left_hand_collision"),
            profile.explicit_pair_names,
        )
        self.assertNotIn(
            ("robot/left_hand_collision", "robot/right_hand_collision"),
            profile.explicit_pair_names,
        )

    def test_wxy_explicit_floor_profile_removes_all_self_pairs(self) -> None:
        floor_leg = get_contact_profile("wxy_explicit_floor_leg_pairs_7caps")
        profile = get_contact_profile("wxy_explicit_floor_pairs_7caps")

        self.assertTrue(profile.eligible_for_parity)
        self.assertEqual(profile.floor_geom_names, ("terrain",))
        self.assertEqual(profile.max_contact_points, floor_leg.max_contact_points)
        self.assertEqual(profile.max_geom_pairs, floor_leg.max_geom_pairs)
        self.assertLess(len(profile.explicit_pair_names), len(floor_leg.explicit_pair_names))
        self.assertIn(
            ("robot/left_foot1_collision", "terrain"),
            profile.explicit_pair_names,
        )
        self.assertIn(
            ("robot/left_hand_collision", "terrain"),
            profile.explicit_pair_names,
        )
        self.assertNotIn(
            ("robot/right_shin_collision", "robot/left_thigh_collision"),
            profile.explicit_pair_names,
        )
        for geom_a, geom_b in profile.explicit_pair_names:
            self.assertIn("terrain", (geom_a, geom_b))
        self.assertEqual(len(profile.explicit_pair_names), 31)

    def test_wxy_pair_param_parity_reuses_floor_leg_pair_set(self) -> None:
        base = get_contact_profile("wxy_explicit_floor_leg_pairs_7caps")
        profile = get_contact_profile(
            "wxy_explicit_floor_leg_pair_param_parity_7caps"
        )

        self.assertTrue(profile.eligible_for_parity)
        self.assertEqual(profile.explicit_pair_names, base.explicit_pair_names)
        self.assertEqual(profile.max_contact_points, base.max_contact_points)
        self.assertEqual(profile.max_geom_pairs, base.max_geom_pairs)

    def test_wxy_floor_leg_foot147_profile_reduces_only_foot_terrain_pairs(
        self,
    ) -> None:
        base = get_contact_profile("wxy_explicit_floor_leg_pairs_7caps")
        profile = get_contact_profile("wxy_explicit_floor_leg_foot147_pairs_7caps")

        self.assertTrue(profile.eligible_for_parity)
        self.assertEqual(profile.floor_geom_names, ("terrain",))
        self.assertEqual(profile.max_contact_points, base.max_contact_points)
        self.assertEqual(profile.max_geom_pairs, base.max_geom_pairs)
        self.assertLess(len(profile.explicit_pair_names), len(base.explicit_pair_names))
        for side in ("left", "right"):
            for index in (1, 4, 7):
                self.assertIn(
                    (f"robot/{side}_foot{index}_collision", "terrain"),
                    profile.explicit_pair_names,
                )
            for index in (2, 3, 5, 6):
                self.assertNotIn(
                    (f"robot/{side}_foot{index}_collision", "terrain"),
                    profile.explicit_pair_names,
                )
        self.assertIn(
            ("robot/left_hand_collision", "terrain"),
            profile.explicit_pair_names,
        )
        self.assertIn(
            ("robot/right_shin_collision", "robot/left_thigh_collision"),
            profile.explicit_pair_names,
        )
        self.assertEqual(len(profile.explicit_pair_names), 36)

    def test_reference_profiles_are_not_parity_eligible(self) -> None:
        self.assertFalse(get_contact_profile("hgpt_track_reference").eligible_for_parity)
        self.assertFalse(get_contact_profile("hgpt_loco_reference").eligible_for_parity)

    @unittest.skipUnless(HGPT_4010_TRACK_SCENE.exists(), "HGPT 4010 assets unavailable")
    def test_hgpt_4010_track_profile_matches_reference_xml(self) -> None:
        profile = get_contact_profile("hgpt_track_reference")
        model = mujoco.MjModel.from_xml_path(str(HGPT_4010_TRACK_SCENE))

        self.assertEqual(model.nq, QPOS_DIM)
        self.assertEqual(model.nv, QVEL_DIM)
        self.assertEqual(model.nu, ACTION_DIM)
        self.assertEqual(model.npair, 57)
        self.assertEqual(profile.floor_geom_names, ("floor",))
        self.assertEqual(profile.explicit_pair_names, _xml_contact_pairs(HGPT_4010_TRACK_SCENE))
        for geom_name in profile.foot_collision_geom_names:
            self.assertGreaterEqual(_geom_id(model, geom_name), 0, geom_name)
        for geom1, geom2 in profile.explicit_pair_names:
            self.assertGreaterEqual(_geom_id(model, geom1), 0, geom1)
            self.assertGreaterEqual(_geom_id(model, geom2), 0, geom2)

        for qpos_index, joint_name in enumerate(MUJOCO_JOINT_NAMES, start=7):
            joint_id = _joint_id(model, joint_name)
            self.assertEqual(model.jnt_qposadr[joint_id], qpos_index, joint_name)
            self.assertEqual(_actuator_name(model, qpos_index - 7), joint_name)
        for body_name in MUJOCO_BODY_NAMES:
            self.assertGreaterEqual(_body_id(model, body_name), 0, body_name)

    @unittest.skipUnless(HGPT_4010_LOCO_SCENE.exists(), "HGPT 4010 assets unavailable")
    def test_hgpt_4010_loco_profile_matches_reference_xml(self) -> None:
        profile = get_contact_profile("hgpt_loco_reference")
        model = mujoco.MjModel.from_xml_path(str(HGPT_4010_LOCO_SCENE))

        self.assertEqual(model.nq, QPOS_DIM)
        self.assertEqual(model.nv, QVEL_DIM)
        self.assertEqual(model.nu, ACTION_DIM)
        self.assertEqual(model.npair, 13)
        self.assertEqual(profile.explicit_pair_names, _xml_contact_pairs(HGPT_4010_LOCO_SCENE))
        for geom_name in profile.foot_collision_geom_names:
            self.assertGreaterEqual(_geom_id(model, geom_name), 0, geom_name)


def _xml_contact_pairs(path: Path) -> tuple[tuple[str, str], ...]:
    root = ET.parse(path).getroot()
    return tuple(
        (str(pair.get("geom1")), str(pair.get("geom2")))
        for pair in root.findall("./contact/pair")
    )


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _actuator_name(model: mujoco.MjModel, index: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
    if name is None:
        raise AssertionError(f"Missing actuator name at index {index}")
    return name


if __name__ == "__main__":
    unittest.main()
