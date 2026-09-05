"""Normalize AudioForge final and release-candidate version strings."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass


_VERSION = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:(?:rc(?P<pep_rc>[1-9][0-9]*))|(?:-rc\.(?P<cargo_rc>[1-9][0-9]*)))?$",
)
_TAG = re.compile(r"^v(?P<version>.+)$")
_MSI_SCALE = 100
_MSI_RC_MAX = _MSI_SCALE - 2
_MSI_PATCH_MAX = (65535 - (_MSI_SCALE - 1)) // _MSI_SCALE
_MSI_MAJOR_MINOR_MAX = 255


@dataclass(frozen=True)
class ReleaseVersion:
    """A release version in the source, tag, Cargo, and MSI namespaces."""

    major: int
    minor: int
    patch: int
    rc: int | None = None

    @property
    def base(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def pep440(self) -> str:
        return self.base if self.rc is None else f"{self.base}rc{self.rc}"

    @property
    def cargo(self) -> str:
        return self.base if self.rc is None else f"{self.base}-rc.{self.rc}"

    @property
    def tag(self) -> str:
        return f"v{self.base}" if self.rc is None else f"v{self.base}-rc.{self.rc}"

    @property
    def is_prerelease(self) -> bool:
        return self.rc is not None

    @property
    def msi(self) -> str:
        """Return a valid MSI version preserving RC-before-final ordering.

        MSI versions have three numeric fields and no prerelease syntax. The
        patch field reserves the final value in each 100-value block for the
        final release; RC sequence numbers occupy the preceding values.
        """

        if self.patch > _MSI_PATCH_MAX:
            raise ValueError(
                f"patch component {self.patch} is too large for the MSI mapping"
            )
        if self.major > _MSI_MAJOR_MINOR_MAX or self.minor > _MSI_MAJOR_MINOR_MAX:
            raise ValueError("major and minor components exceed the MSI limit (255)")
        if self.rc is not None and self.rc > _MSI_RC_MAX:
            raise ValueError(
                f"release-candidate sequence {self.rc} exceeds the MSI limit "
                f"({_MSI_RC_MAX})"
            )
        patch = self.patch * _MSI_SCALE + (
            self.rc if self.rc is not None else _MSI_SCALE - 1
        )
        return f"{self.major}.{self.minor}.{patch}"


def parse_version(value: str) -> ReleaseVersion:
    """Parse a final or RC source version, rejecting ambiguous spellings."""

    match = _VERSION.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            "version must be MAJOR.MINOR.PATCH, 2.0.0rc1, or 2.0.0-rc.1"
        )
    groups = match.groupdict()
    rc_values = [groups["pep_rc"], groups["cargo_rc"]]
    return ReleaseVersion(
        int(groups["major"]),
        int(groups["minor"]),
        int(groups["patch"]),
        int(next(value for value in rc_values if value is not None))
        if any(value is not None for value in rc_values)
        else None,
    )


def parse_tag(value: str) -> ReleaseVersion:
    """Parse the canonical immutable tag spelling used by workflows."""

    match = _TAG.fullmatch(value.strip())
    if match is None:
        raise ValueError("release tag must start with v")
    parsed = parse_version(match.group("version"))
    if parsed.rc is not None and parsed.tag != value.strip():
        raise ValueError(f"release candidate tag must use {parsed.tag}")
    if parsed.rc is None and parsed.tag != value.strip():
        raise ValueError(f"release tag must use {parsed.tag}")
    return parsed


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("field", choices=("tag", "cargo", "pep440", "msi", "prerelease"))
    parser.add_argument("version")
    args = parser.parse_args()
    value = (
        parse_tag(args.version)
        if args.version.startswith("v")
        else parse_version(args.version)
    )
    if args.field == "prerelease":
        print(str(value.is_prerelease).lower())
    else:
        print(getattr(value, args.field))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except ValueError as exc:
        parser_error = argparse.ArgumentParser(prog="release_version")
        parser_error.error(str(exc))
