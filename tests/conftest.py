"""Shared pytest fixtures.

GPU tests are separated from pure logic tests so the suite still runs somewhere
without a usable EGL device: anything needing a context is marked ``gpu`` and
skipped automatically when no hardware is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires a working GL context")


@pytest.fixture(scope="session")
def gl_context():
    """A single shared context; creating one per test would be wasteful."""
    from shadertoy_local.context import ContextError, create_context

    try:
        handle = create_context()
    except ContextError as exc:
        pytest.skip(f"no usable GPU context: {exc}")
    yield handle
    handle.release()


@pytest.fixture
def make_project(tmp_path: Path):
    """Build a throwaway project directory from a dict of files."""

    def _make(files: dict[str, str], config: dict | None = None) -> Path:
        root = tmp_path / "proj"
        root.mkdir(exist_ok=True)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        if config is not None:
            (root / "shadertoy.json").write_text(
                json.dumps(config, indent=2), encoding="utf-8"
            )
        return root

    return _make


#: A trivial pass that compiles and produces a non-uniform frame.
SIMPLE_IMAGE = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = vec4(uv, 0.5, 1.0);
}
"""

#: Left half red, right half blue -- exact values, ideal for probe assertions.
HALVES_IMAGE = """
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = fragCoord.x < iResolution.x * 0.5
        ? vec4(1.0, 0.0, 0.0, 1.0)
        : vec4(0.0, 0.0, 1.0, 1.0);
}
"""


@pytest.fixture
def simple_image() -> str:
    return SIMPLE_IMAGE


@pytest.fixture
def halves_image() -> str:
    return HALVES_IMAGE
