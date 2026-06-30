#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from qselect_debug_eval import build_policy_element


class DebugEvalRequestTest(unittest.TestCase):
    def test_policy_element_contains_rollout_identifiers(self) -> None:
        processed = {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "wrist_image": np.ones((2, 2, 3), dtype=np.uint8),
            "state": np.arange(8, dtype=np.float32),
        }

        element = build_policy_element(
            processed,
            "pick up the bowl",
            episode_id=7,
            env_step=20,
            task_id=3,
        )

        self.assertEqual(element["episode_id"], 7)
        self.assertEqual(element["env_step"], 20)
        self.assertEqual(element["task_id"], 3)
        self.assertEqual(element["prompt"], "pick up the bowl")
        np.testing.assert_array_equal(element["observation/state"], processed["state"])


if __name__ == "__main__":
    unittest.main()
