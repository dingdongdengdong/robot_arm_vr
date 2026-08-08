"""WebXR and dashboard rendering defaults."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quest_uses_real_urdf_meshes_by_default() -> None:
    page = (ROOT / "web" / "index.html").read_text()

    assert 'get("mesh") !== "0"' in page
    assert "?mesh=0" in page


def test_dashboard_uses_real_urdf_meshes_by_default() -> None:
    page = (ROOT / "web" / "dashboard.html").read_text()

    assert 'get("mesh") !== "0"' in page


def test_quest_keeps_the_original_virtual_stage_direction() -> None:
    page = (ROOT / "web" / "index.html").read_text()

    assert "STAGE_YAW_DEG" not in page
    assert "stage.rotation.set" not in page
    assert "if (lg.stickPress) resetStage();" in page
