import re
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]


def _optional_dependencies() -> dict[str, list[str]]:
    content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section_match = re.search(
        r"(?ms)^\[project\.optional-dependencies\]\n(.*?)(?=^\[)",
        content,
    )
    assert section_match is not None, "Missing [project.optional-dependencies] section"

    section = section_match.group(1)
    extras: dict[str, list[str]] = {}
    for match in re.finditer(r"(?ms)^([A-Za-z0-9_-]+)\s*=\s*\[(.*?)^\]", section):
        name = match.group(1)
        values = re.findall(r'"([^"]+)"', match.group(2))
        extras[name] = values
    return extras


def _requirement_names(requirements: list[str]) -> set[str]:
    return {Requirement(requirement).name.lower() for requirement in requirements}


def test_all_extra_covers_every_runtime_extra() -> None:
    extras = _optional_dependencies()
    expected_names = _requirement_names(
        extras["viz"]
        + extras["torch"]
        + extras["tf"]
        + extras["infer-tokenizers"]
        + extras["tui"]
        + extras["jax"]
        + extras["wandb"]
    )

    assert expected_names.issubset(_requirement_names(extras["all"])), (
        "The all extra must cover every user-facing runtime extra "
        "(viz, torch, tf, infer-tokenizers, tui, jax, wandb)."
    )


def test_all_extra_uses_protobuf_compatible_tensorflow() -> None:
    extras = _optional_dependencies()

    assert "tensorflow>=2.21.0" in extras["all"]
    assert "protobuf>=6.31.1" in extras["all"]
