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


CONTACT_PROFILES: dict[str, ContactProfile] = {
    "wxy_parity": ContactProfile(
        name="wxy_parity",
        floor_geom_names=("terrain", "floor"),
        foot_body_names=("robot/left_ankle_roll_link", "robot/right_ankle_roll_link"),
        foot_collision_geom_names=WXY_FOOT_COLLISION_GEOMS,
        explicit_pair_names=tuple(("terrain", geom) for geom in WXY_FOOT_COLLISION_GEOMS),
        max_contact_points=512,
        max_geom_pairs=1024,
        eligible_for_parity=True,
        notes="Preserves WXY seven-capsule foot collision semantics.",
    ),
    "hgpt_track_reference": ContactProfile(
        name="hgpt_track_reference",
        floor_geom_names=("floor",),
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        foot_collision_geom_names=(
            "left_foot_box",
            "left_foot_capsule_1",
            "left_foot_capsule_2",
            "left_foot_capsule_3",
            "right_foot_box",
            "right_foot_capsule_1",
            "right_foot_capsule_2",
            "right_foot_capsule_3",
        ),
        explicit_pair_names=(),
        max_contact_points=128,
        max_geom_pairs=256,
        eligible_for_parity=False,
        notes="Humanoid-GPT tracking reference; changes WXY foot geometry.",
    ),
    "hgpt_loco_reference": ContactProfile(
        name="hgpt_loco_reference",
        floor_geom_names=("floor",),
        foot_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        foot_collision_geom_names=(
            "left_foot_box",
            "right_foot_box",
        ),
        explicit_pair_names=(),
        max_contact_points=64,
        max_geom_pairs=128,
        eligible_for_parity=False,
        notes="Humanoid-GPT locomotion reference; reduced contact set.",
    ),
}


def get_contact_profile(name: str) -> ContactProfile:
    """Return a named contact profile or raise a clear error."""

    try:
        return CONTACT_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(CONTACT_PROFILES))
        raise ValueError(f"Unknown MJX contact profile {name!r}; expected one of {known}.") from exc
