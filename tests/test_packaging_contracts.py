import re
from pathlib import Path

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


def test_all_extra_covers_every_runtime_extra() -> None:
    extras = _optional_dependencies()
    expected = (
        set(extras["viz"])
        | set(extras["torch"])
        | set(extras["tf"])
        | set(extras["tui"])
    )

    assert expected.issubset(set(extras["all"])), (
        "The all extra must cover every user-facing runtime extra "
        "(viz, torch, tf, tui)."
    )
