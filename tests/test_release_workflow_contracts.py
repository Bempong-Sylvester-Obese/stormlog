from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_uses_manual_dispatch_and_trusted_publishing() -> None:
    content = (REPO_ROOT / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in content
    assert "workflow_run:" not in content
    assert "environment: pypi" in content
    assert "id-token: write" in content
    assert "pypa/gh-action-pypi-publish@" in content
    assert "PYPI_API_TOKEN" not in content
    assert "twine upload" not in content
    assert "gh release create" in content
    assert "persist-credentials: false" in content
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" in content
