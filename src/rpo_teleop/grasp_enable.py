"""Quest 2 mapping for AmazingHand (home/A) and tongs (index trigger).

Right A (``buttons[4]``, the Quest 2 home/A button) opens or closes
AmazingHand. The right index trigger is the analog tongs end-effector.
"""

from __future__ import annotations


def toggle_on_rising_edge(on: bool, pressed: bool, was_pressed: bool) -> tuple[bool, bool]:
    """Flip a latch on the rising edge of a Quest button."""
    if pressed and not was_pressed:
        on = not on
    return bool(on), bool(pressed)


def amazing_hand_grasp(closed: bool) -> float:
    """AmazingHand command from the home/A latch. 0=open, 1=closed."""
    return 1.0 if closed else 0.0


def tongs_grasp(trigger: float) -> float:
    """Tongs end-effector command from the Quest right index trigger."""
    return min(1.0, max(0.0, float(trigger)))


def amazing_hand_amount(command) -> float | None:
    """Pick the AmazingHand 0..1 scalar from a Jetson command.

    ``hand_grasp`` is the home/A button. ``grasp`` is the tongs trigger and must
    not move AmazingHand when ``hand_grasp`` is present. Legacy packets that
    only carry ``grasp`` still drive the hand.
    """
    hand_grasp = getattr(command, "hand_grasp", None)
    if hand_grasp is not None:
        return tongs_grasp(hand_grasp)
    if getattr(command, "servo", None) is not None:
        return None
    grasp = getattr(command, "grasp", None)
    if grasp is None:
        return None
    return tongs_grasp(grasp)
