"""Quest 2 right A/home button on/off gate for AmazingHand grasping.

The analog trigger still sets the grasp amount. The home button (Quest 2
right A, ``buttons[4]``) only enables or disables that mapping.
"""

from __future__ import annotations


def toggle_grasp_enable(enabled: bool, pressed: bool, was_pressed: bool) -> tuple[bool, bool]:
    """Flip grasp-enable on the rising edge of the Quest right home/A button."""
    if pressed and not was_pressed:
        enabled = not enabled
    return bool(enabled), bool(pressed)


def gated_grasp(trigger: float, enabled: bool, held: float) -> float:
    """Follow the analog trigger while enabled; otherwise keep the last grasp."""
    amount = min(1.0, max(0.0, float(trigger)))
    if enabled:
        return amount
    return min(1.0, max(0.0, float(held)))
