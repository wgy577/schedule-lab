"""Apply and audit reusable A-H test tags from one canonical registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


TEST_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = TEST_ROOT / "test_registry.json"


def _registry() -> dict[str, dict[str, object]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["tests"]


def _markers(metadata: dict[str, object]) -> tuple[str, ...]:
    modules = tuple(f"module_{item.lower()}" for item in metadata["modules"])
    capabilities = tuple(f"cap_{item}" for item in metadata["capabilities"])
    return (*modules, *capabilities, f"level_{metadata['level']}", f"cost_{metadata['cost']}")


def _expected_header(metadata: dict[str, object]) -> str:
    return (
        "# TEST-TAGS: "
        f"modules={','.join(metadata['modules'])}; "
        f"capabilities={','.join(metadata['capabilities'])}; "
        f"level={metadata['level']}; cost={metadata['cost']}"
    )


def pytest_configure(config: pytest.Config) -> None:
    registry = _registry()
    descriptions: dict[str, str] = {}
    for filename, metadata in registry.items():
        for marker in _markers(metadata):
            descriptions.setdefault(marker, f"registered by tests/test_registry.json ({filename})")
    for marker, description in sorted(descriptions.items()):
        config.addinivalue_line("markers", f"{marker}: {description}")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    registry = _registry()
    discovered = {path.name for path in TEST_ROOT.glob("test_*.py")}
    missing = sorted(discovered - set(registry))
    stale = sorted(set(registry) - discovered)
    if missing or stale:
        raise pytest.UsageError(
            f"test registry mismatch; missing={missing}, stale={stale}"
        )
    for item in items:
        filename = Path(str(item.path)).name
        metadata = registry.get(filename)
        if metadata is None:
            continue
        source = (TEST_ROOT / filename).read_text(encoding="utf-8")
        first_line = source.splitlines()[0] if source else ""
        expected_header = _expected_header(metadata)
        if first_line != expected_header:
            raise pytest.UsageError(
                f"{filename} TEST-TAGS mismatch; expected: {expected_header}"
            )
        for marker in _markers(metadata):
            item.add_marker(getattr(pytest.mark, marker))
