from __future__ import annotations

import pytest

from stormlog.release_version import (
    DEFAULT_INITIAL_VERSION,
    PATCH_ROLLOVER,
    bump_release_version,
    resolve_release_version,
    split_release_tags,
)


def test_split_release_tags_separates_stable_and_prerelease_tags() -> None:
    stable, prerelease = split_release_tags(
        [
            "v0.2.9",
            "v0.2.10",
            "v0.3.0rc1",
            "v0.3.0.post1",
            "not-a-version",
        ]
    )

    assert stable == [(0, 2, 9), (0, 2, 10)]
    assert prerelease == [(0, 3, 0), (0, 3, 0)]


def test_bump_release_version_preserves_patch_until_rollover() -> None:
    assert bump_release_version((0, 2, 9)) == (0, 2, 10)


def test_bump_release_version_rolls_minor_after_patch_ten() -> None:
    assert bump_release_version((0, 2, PATCH_ROLLOVER)) == (0, 3, 0)


def test_bump_release_version_rejects_negative_rollover() -> None:
    with pytest.raises(ValueError, match="patch_rollover must be >= 0"):
        bump_release_version((0, 2, 0), patch_rollover=-1)


def test_resolve_release_version_uses_latest_stable_tag() -> None:
    version = resolve_release_version(["v0.2.8", "v0.2.10", "v0.3.0rc1"])

    assert version == "0.3.0"


def test_resolve_release_version_uses_prerelease_base_when_no_stable_exists() -> None:
    version = resolve_release_version(["v0.3.0rc1", "v0.2.10a1"])

    assert version == "0.3.0"


def test_resolve_release_version_uses_default_when_no_release_tags_exist() -> None:
    version = resolve_release_version(["junk-tag"])

    assert version == DEFAULT_INITIAL_VERSION
