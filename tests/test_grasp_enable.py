from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpo_teleop.grasp_enable import (  # noqa: E402
    amazing_hand_amount,
    amazing_hand_grasp,
    tongs_grasp,
    toggle_on_rising_edge,
)


def test_home_button_toggles_amazing_hand_open_and_closed() -> None:
    closed, was_pressed = False, False

    closed, was_pressed = toggle_on_rising_edge(closed, pressed=True, was_pressed=was_pressed)
    assert amazing_hand_grasp(closed) == 1.0

    closed, was_pressed = toggle_on_rising_edge(closed, pressed=False, was_pressed=was_pressed)
    closed, was_pressed = toggle_on_rising_edge(closed, pressed=True, was_pressed=was_pressed)
    assert amazing_hand_grasp(closed) == 0.0


def test_index_trigger_is_the_tongs_end_effector() -> None:
    assert tongs_grasp(0.35) == 0.35
    assert tongs_grasp(9.0) == 1.0


def test_home_button_hand_grasp_is_not_the_tongs_trigger() -> None:
    command = SimpleNamespace(hand_grasp=1.0, grasp=0.3, servo=None)
    assert amazing_hand_amount(command) == 1.0

    legacy = SimpleNamespace(grasp=0.75)
    assert amazing_hand_amount(legacy) == 0.75

    tongs_only = SimpleNamespace(hand_grasp=0.0, grasp=0.9, servo=[0.1] * 8)
    assert amazing_hand_amount(tongs_only) == 0.0

    servo_only = SimpleNamespace(grasp=0.9, servo=[0.1] * 8)
    assert amazing_hand_amount(servo_only) is None

    missing = SimpleNamespace(servo=None, grasp=None)
    assert amazing_hand_amount(missing) is None
