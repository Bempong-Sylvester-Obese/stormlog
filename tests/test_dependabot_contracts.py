from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dependabot_enables_cooldowns_for_pip_and_actions() -> None:
    content = (REPO_ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")

    assert 'package-ecosystem: "pip"' in content
    assert 'package-ecosystem: "github-actions"' in content
    assert "cooldown:" in content
    assert "default-days: 7" in content
