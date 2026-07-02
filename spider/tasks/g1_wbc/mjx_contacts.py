"""MJX contact profile manifests for G1 WBC model variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactProfile:
    name: str
    floor_geom_names: tuple[str, ...]
    foot_body_names: tuple[str, ...]
    foot_collision_geom_names: tuple[str, ...]
    explicit_pair_names: tuple[tuple[str, str], ...]
    max_contact_points: int
    max_geom_pairs: int
    eligible_for_parity: bool
    notes: str


WXY_FOOT_COLLISION_GEOMS = tuple(
    f"robot/{side}_foot{index}_collision"
    for side in ("left", "right")
    for index in range(1, 8)
)

HGPT_4010_FOOT_COLLISION_GEOMS = (
    "left_foot_box_collision",
    "left_foot1_collision",
    "left_foot2_collision",
    "left_foot3_collision",
    "right_foot_box_collision",
    "right_foot1_collision",
    "right_foot2_collision",
    "right_foot3_collision",
)

HGPT_4010_LOCO_EXPLICIT_PAIRS = (
    ("left_foot1_collision", "floor"),
    ("left_foot2_collision", "floor"),
    ("left_foot3_collision", "floor"),
    ("right_foot1_collision", "floor"),
    ("right_foot2_collision", "floor"),
    ("right_foot3_collision", "floor"),
    ("left_foot_box_collision", "right_foot_box_collision"),
    ("left_foot_box_collision", "right_shin_collision"),
    ("right_foot_box_collision", "left_shin_collision"),
    ("left_foot_box_collision", "right_linkage_brace_collision"),
    ("right_foot_box_collision", "left_linkage_brace_collision"),
    ("left_shin_collision", "right_shin_collision"),
    ("left_linkage_brace_collision", "right_linkage_brace_collision"),
)

HGPT_4010_TRACK_EXPLICIT_PAIRS = (
    ("left_foot1_collision", "floor"),
    ("left_foot2_collision", "floor"),
    ("left_foot3_collision", "floor"),
    ("right_foot1_collision", "floor"),
    ("right_foot2_collision", "floor"),
    ("right_foot3_collision", "floor"),
    ("left_foot_box_collision", "right_foot_box_collision"),
    ("left_foot_box_collision", "right_shin_collision"),
    ("right_foot_box_collision", "left_shin_collision"),
    ("left_foot_box_collision", "right_linkage_brace_collision"),
    ("right_foot_box_collision", "left_linkage_brace_collision"),
    ("left_hand_collision", "left_hip_collision"),
    ("right_hand_collision", "right_hip_collision"),
    ("left_hand_collision", "left_thigh_collision"),
    ("right_hand_collision", "right_thigh_collision"),
    ("left_shin_collision", "right_shin_collision"),
    ("torso_collision", "left_shoulder_yaw_collision"),
    ("torso_collision", "right_shoulder_yaw_collision"),
    ("torso_collision", "left_elbow_yaw_collision"),
    ("torso_collision", "right_elbow_yaw_collision"),
    ("torso_collision", "left_wrist_collision"),
    ("torso_collision", "right_wrist_collision"),
    ("torso_collision", "left_hand_collision"),
    ("torso_collision", "right_hand_collision"),
    ("pelvis_collision", "left_wrist_collision"),
    ("pelvis_collision", "right_wrist_collision"),
    ("pelvis_collision", "left_hand_collision"),
    ("pelvis_collision", "right_hand_collision"),
    ("left_hand_collision", "floor"),
    ("right_hand_collision", "floor"),
    ("left_shoulder_yaw_collision", "floor"),
    ("right_shoulder_yaw_collision", "floor"),
    ("left_elbow_yaw_collision", "floor"),
    ("right_elbow_yaw_collision", "floor"),
    ("left_wrist_collision", "floor"),
    ("right_wrist_collision", "floor"),
    ("left_hip_collision", "floor"),
    ("right_hip_collision", "floor"),
    ("left_thigh_collision", "floor"),
    ("right_thigh_collision", "floor"),
    ("left_thigh_collision", "left_hand_collision"),
    ("right_thigh_collision", "right_hand_collision"),
    ("left_shin_collision", "floor"),
    ("right_shin_collision", "floor"),
    ("pelvis_collision", "floor"),
    ("torso_collision", "floor"),
    ("head_collision", "floor"),
    ("head_collision", "left_hand_collision"),
    ("head_collision", "right_hand_collision"),
    ("left_thigh_collision", "right_thigh_collision"),
    ("left_shin_collision", "right_thigh_collision"),
    ("right_shin_collision", "left_thigh_collision"),
    ("left_shin_collision", "right_hip_collision"),
    ("right_shin_collision", "left_hip_collision"),
    ("left_hip_collision", "right_thigh_collision"),
    ("right_hip_collision", "left_thigh_collision"),
    ("left_hand_collision", "right_hand_collision"),
)


CONTACT_PROFILES: dict[str, ContactProfile] = {
    "wxy_parity": ContactProfile(
        name="wxy_parity",
        floor_geom_names=("terrain",),
        foot_body_names=("robot/left_ankle_roll_link", "robot/right_ankle_roll_link"),
        foot_collision_geom_names=WXY_FOOT_COLLISION_GEOMS,
        explicit_pair_names=(),
        max_contact_points=512,
        max_geom_pairs=1024,
        eligible_for_parity=True,
        notes="Preserves WXY seven-capsule foot collision semantics in the CPU bundle.",
    ),
    "hgpt_track_reference": ContactProfile(
        name="hgpt_track_reference",
        floor_geom_names=("floor",),
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        foot_collision_geom_names=HGPT_4010_FOOT_COLLISION_GEOMS,
        explicit_pair_names=HGPT_4010_TRACK_EXPLICIT_PAIRS,
        max_contact_points=128,
        max_geom_pairs=256,
        eligible_for_parity=False,
        notes="Humanoid-GPT 4010 tracking reference; changes WXY foot geometry.",
    ),
    "hgpt_loco_reference": ContactProfile(
        name="hgpt_loco_reference",
        floor_geom_names=("floor",),
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        foot_collision_geom_names=HGPT_4010_FOOT_COLLISION_GEOMS,
        explicit_pair_names=HGPT_4010_LOCO_EXPLICIT_PAIRS,
        max_contact_points=64,
        max_geom_pairs=128,
        eligible_for_parity=False,
        notes="Humanoid-GPT 4010 locomotion reference; reduced contact set.",
    ),
}


def get_contact_profile(name: str) -> ContactProfile:
    """Return a named contact profile or raise a clear error."""

    try:
        return CONTACT_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(CONTACT_PROFILES))
        raise ValueError(f"Unknown MJX contact profile {name!r}; expected one of {known}.") from exc
