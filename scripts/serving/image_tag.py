"""The serving image's tag: a content hash of everything it is built from.

The serving image (the Dockerfile's `runtime` target without the extras) is
tagged with a hash of its inputs -- the files the Dockerfile copies into it,
the Dockerfile and .dockerignore, and the build arguments that change what's
in it -- not with a commit: a commit hash covers the whole tree and isn't
known before the commit is made, while this is. So the deployment can pin
the tag in the same pull request that changes the code, and CI builds and
pushes the image under it on that pull request, before it merges, never
overwriting a tag: the image that deploys is the one CI built and tested
from exactly those inputs (README, "Image publishing").

The hash is over git's index (`git ls-files -s`: the mode, blob hash and
path of every tracked input file), so it's what is committed, not the
working tree. Standard library only: CI runs it before any environment is
set up.

`--pin` writes the tag into a Kustomize overlay (its `newTag` for this
repository): the pre-commit hook does, for the overlay Argo CD deploys, and
a test checks the committed pin, so the deployment never names an image
other than the one built from its own commit's inputs.

Usage:
    python3 scripts/serving/image_tag.py                    # the image reference
    python3 scripts/serving/image_tag.py --github-output    # key=value lines for CI
    python3 scripts/serving/image_tag.py --pin deploy/k8s/overlays/kind/kustomization.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

REPOSITORY = "ghcr.io/michaelskowronek/mhd-surrogate-serve"
# What the image is built from: the sources of the Dockerfile's COPY and
# bind-mount instructions up to the runtime target (checked by
# tests/serving/test_image_tag.py), and the build files themselves.
INPUTS = (
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "uv.lock",
    "src",
    "scripts",
    "configs",
)
# The build: its target and the build arguments that change the image's
# content (GIT_COMMIT only labels it).
BUILD = {"target": "runtime", "UV_EXTRAS": ""}
TAG_LENGTH = 16
# The overlay's image entry: `newName: <REPOSITORY>` and, on the next line,
# its `newTag`, quoted (an unquoted hex tag can read as a YAML number).
PIN = re.compile(rf'(newName: {re.escape(REPOSITORY)}\n\s*newTag: )"?(\w*)"?')


def inputs_listing(root: Path | str = ".") -> str:
    """`git ls-files -s` of the inputs in the repository at `root`."""
    return subprocess.run(
        ["git", "ls-files", "-s", "--", *INPUTS],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def image_tag(listing: str, build: Mapping[str, str] = BUILD) -> str:
    """The tag of an image built as `build` from the inputs `listing`."""
    digest = hashlib.sha256(json.dumps(dict(build), sort_keys=True).encode())
    digest.update(listing.encode())
    return digest.hexdigest()[:TAG_LENGTH]


def pinned_tag(text: str) -> str:
    """The tag a Kustomize overlay's text pins for the serving image."""
    match = PIN.search(text)
    if match is None:
        raise ValueError(f"no newTag for {REPOSITORY} in the overlay")
    return match.group(2)


def pin(path: Path, tag: str) -> bool:
    """Pin the overlay at `path` to `tag`; whether that changed it."""
    text = path.read_text()
    pinned_tag(text)
    pinned = PIN.sub(lambda match: f'{match.group(1)}"{tag}"', text, count=1)
    if pinned == text:
        return False
    path.write_text(pinned)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--github-output", action="store_true")
    parser.add_argument("--pin", type=Path, metavar="KUSTOMIZATION")
    args = parser.parse_args()
    tag = image_tag(inputs_listing())
    if args.pin:
        if pin(args.pin, tag):
            sys.exit(f"pinned {args.pin} to {tag}: stage it with the change")
    elif args.github_output:
        print(f"image={REPOSITORY}:{tag}")
        print(f"target={BUILD['target']}")
        print(f"uv_extras={BUILD['UV_EXTRAS']}")
    else:
        print(f"{REPOSITORY}:{tag}")


if __name__ == "__main__":
    main()
