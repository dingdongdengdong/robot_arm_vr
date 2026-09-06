from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpo_teleop.grasp_enable import gated_grasp, toggle_grasp_enable  # noqa: E402


def test_home_button_rising_edge_toggles_amazing_hand_grasp_enable() -> None:
    enabled, was_pressed = True, False

    enabled, was_pressed = toggle_grasp_enable(enabled, pressed=False, was_pressed=was_pressed)
    assert enabled is True

    enabled, was_pressed = toggle_grasp_enable(enabled, pressed=True, was_pressed=was_pressed)
    assert enabled is False
    assert was_pressed is True

    enabled, was_pressed = toggle_grasp_enable(enabled, pressed=True, was_pressed=was_pressed)
    assert enabled is False

    enabled, was_pressed = toggle_grasp_enable(enabled, pressed=False, was_pressed=was_pressed)
    enabled, was_pressed = toggle_grasp_enable(enabled, pressed=True, was_pressed=was_pressed)
    assert enabled is True


def test_disabled_grasp_holds_last_amount_and_ignores_trigger() -> None:
    held = gated_grasp(0.8, enabled=True, held=0.0)
    assert held == 0.8
    assert gated_grasp(0.1, enabled=False, held=held) == 0.8
    assert gated_grasp(1.0, enabled=True, held=held) == 1.0
