"""URDF 작업공간 파생값 회귀 테스트."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rpo_teleop.arm_config import safe_reach_bounds  # noqa: E402


def test_safe_reach_bounds_keeps_normal_margin() -> None:
    lo, hi = safe_reach_bounds(np.linspace(0.10, 0.50, 1000), 0.86)
    assert hi == pytest.approx(0.43)
    assert 0.10 < lo < hi


def test_safe_reach_bounds_never_inverts_thin_two_dof_shell() -> None:
    reach = np.linspace(0.458, 0.500, 1000)
    lo, hi = safe_reach_bounds(reach, 0.86)
    assert lo < hi
    assert hi == pytest.approx(0.500)


@pytest.mark.parametrize("reach, margin", [([], 0.86), ([np.nan], 0.86), ([1.0], 0.0)])
def test_safe_reach_bounds_rejects_invalid_input(reach, margin) -> None:
    with pytest.raises(ValueError):
        safe_reach_bounds(np.asarray(reach), margin)
