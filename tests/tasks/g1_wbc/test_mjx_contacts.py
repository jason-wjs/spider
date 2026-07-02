import unittest

from spider.tasks.g1_wbc.mjx_contacts import CONTACT_PROFILES, get_contact_profile


class MjxContactProfilesTest(unittest.TestCase):
    def test_contact_profiles_include_wxy_and_hgpt_references(self) -> None:
        self.assertEqual(
            set(CONTACT_PROFILES),
            {"wxy_parity", "hgpt_track_reference", "hgpt_loco_reference"},
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

    def test_reference_profiles_are_not_parity_eligible(self) -> None:
        self.assertFalse(get_contact_profile("hgpt_track_reference").eligible_for_parity)
        self.assertFalse(get_contact_profile("hgpt_loco_reference").eligible_for_parity)


if __name__ == "__main__":
    unittest.main()
