"""Helpers for deriving the next release version from Git tags."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from typing import cast

PATCH_ROLLOVER = 10
DEFAULT_INITIAL_VERSION = "0.1.0"

VersionTuple = tuple[int, int, int]

_STABLE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_PRERELEASE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:[A-Za-z0-9.+-].*)$")


def _parse_version_tuple(tag: str, pattern: re.Pattern[str]) -> VersionTuple | None:
    match = pattern.fullmatch(tag)
    if match is None:
        return None
    return cast(VersionTuple, tuple(int(part) for part in match.groups()))


def split_release_tags(
    tags: Iterable[str],
) -> tuple[list[VersionTuple], list[VersionTuple]]:
    """Return stable and prerelease version tuples parsed from Git tags."""
    stable_versions: list[VersionTuple] = []
    prerelease_bases: list[VersionTuple] = []

    for tag in tags:
        stable = _parse_version_tuple(tag, _STABLE_TAG_RE)
        if stable is not None:
            stable_versions.append(stable)
            continue

        prerelease = _parse_version_tuple(tag, _PRERELEASE_TAG_RE)
        if prerelease is not None:
            prerelease_bases.append(prerelease)

    return stable_versions, prerelease_bases


def bump_release_version(
    version: VersionTuple,
    *,
    patch_rollover: int = PATCH_ROLLOVER,
) -> VersionTuple:
    """Increment a stable version, rolling patch values above the threshold."""
    if patch_rollover < 0:
        raise ValueError("patch_rollover must be >= 0")

    major, minor, patch = version
    next_patch = patch + 1
    if next_patch > patch_rollover:
        return (major, minor + 1, 0)
    return (major, minor, next_patch)


def format_release_version(version: VersionTuple) -> str:
    """Render a version tuple as a dotted version string."""
    major, minor, patch = version
    return f"{major}.{minor}.{patch}"


def resolve_release_version(
    tags: Iterable[str],
    *,
    patch_rollover: int = PATCH_ROLLOVER,
    default_version: str = DEFAULT_INITIAL_VERSION,
) -> str:
    """Resolve the next stable release version from existing Git tags."""
    stable_versions, prerelease_bases = split_release_tags(tags)

    if stable_versions:
        latest = max(stable_versions)
        return format_release_version(
            bump_release_version(latest, patch_rollover=patch_rollover)
        )

    if prerelease_bases:
        return format_release_version(max(prerelease_bases))

    return default_version


def git_release_tags() -> list[str]:
    """Return Git tags that participate in release-version resolution."""
    return subprocess.check_output(
        ["git", "tag", "--list", "v*"],
        text=True,
    ).splitlines()


def main() -> None:
    """Print the next release version for the current Git checkout."""
    print(resolve_release_version(git_release_tags()))


if __name__ == "__main__":
    main()
