import re
import shutil
import subprocess
from pathlib import Path

import pytest
from image_tag import BUILD, INPUTS, REPOSITORY, image_tag, inputs_listing, pin, pinned_tag

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "deploy/k8s/overlays/kind/kustomization.yaml"

# The Docker test image has neither git nor the Dockerfile.
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="no git")
needs_dockerfile = pytest.mark.skipif(
    not (ROOT / "Dockerfile").exists(), reason="no Dockerfile (the image has none)"
)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def write(repo: Path, path: str, text: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    for path in ["Dockerfile", "README.md", "src/a.py", "configs/c.yaml", "tests/test_a.py"]:
        write(tmp_path, path, path)
    git(tmp_path, "add", "-A")
    return tmp_path


def tag(repo: Path) -> str:
    return image_tag(inputs_listing(repo))


@needs_git
def test_the_tag_changes_with_an_input_and_only_with_an_input(repo):
    before = tag(repo)

    write(repo, "tests/test_a.py", "changed")
    write(repo, "notes.txt", "new")
    git(repo, "add", "-A")
    assert tag(repo) == before

    write(repo, "src/a.py", "changed")
    git(repo, "add", "-A")
    assert tag(repo) != before


@needs_git
def test_the_tag_is_of_the_index_not_the_working_tree(repo):
    before = tag(repo)

    write(repo, "src/a.py", "not staged")
    write(repo, "src/b.py", "untracked")

    assert tag(repo) == before


@needs_git
def test_the_build_arguments_are_part_of_the_tag(repo):
    listing = inputs_listing(repo)

    assert image_tag(listing) == image_tag(listing, dict(BUILD))
    assert image_tag(listing) != image_tag(listing, {**BUILD, "UV_EXTRAS": "--extra ray"})
    assert len(image_tag(listing)) == 16


def dockerfile_sources(text: str, target: str) -> set[str]:
    """What the Dockerfile's `target` stage takes from the build context:
    its COPY and bind-mount sources and those of the stages it is built on
    or copies from."""
    stages: dict[str, tuple[set[str], set[str]]] = {}
    current = None
    for line in re.sub(r"\\\n", " ", text).splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        if words[0].upper() == "FROM":
            name = words[3].lower() if len(words) > 3 else f"stage-{len(stages)}"
            current = stages[name] = (set(), {words[1].lower()})
            continue
        if current is None:
            continue
        if words[0].upper() == "COPY":
            copied_from = [w.split("=", 1)[1].lower() for w in words if w.startswith("--from=")]
            if copied_from:
                current[1].update(copied_from)
            else:
                current[0].update(w for w in words[1:-1] if not w.startswith("--"))
        current[0].update(re.findall(r"--mount=type=bind,source=([^,\s]+)", line))
    sources, todo, seen = set(), [target], set()
    while todo:
        stage = todo.pop()
        if stage in seen or stage not in stages:
            continue
        seen.add(stage)
        sources |= stages[stage][0]
        todo.extend(stages[stage][1])
    return sources


@needs_dockerfile
def test_the_inputs_are_what_the_dockerfile_copies_into_the_runtime_image():
    sources = dockerfile_sources((ROOT / "Dockerfile").read_text(), BUILD["target"])

    assert sources == {"pyproject.toml", "uv.lock", "README.md", "src", "scripts", "configs"}
    assert sources <= set(INPUTS)


def test_dockerfile_sources_follows_from_and_copy_from():
    text = """ARG X=1
FROM python AS base
RUN --mount=type=bind,source=lock.txt,target=lock.txt \\
    install
COPY a.txt ./
FROM base AS build
COPY --chown=app:app b ./b
FROM python AS final
COPY --from=build /app /app
COPY --from=ghcr.io/x/y:1 /bin/y /bin/y
FROM final AS other
COPY tests ./tests
"""

    assert dockerfile_sources(text, "final") == {"lock.txt", "a.txt", "b"}
    assert dockerfile_sources(text, "other") == {"lock.txt", "a.txt", "b", "tests"}


OVERLAY_TEXT = f"""images:
  - name: other
    newName: ghcr.io/x/other
    newTag: "1111"
  - name: mhd-surrogate-serve
    newName: {REPOSITORY}
    newTag: "0000000000000000"
"""


def test_pin_rewrites_the_serving_images_tag_once(tmp_path):
    path = tmp_path / "kustomization.yaml"
    path.write_text(OVERLAY_TEXT)

    assert pin(path, "abcdef0123456789")
    assert pinned_tag(path.read_text()) == "abcdef0123456789"
    assert path.read_text() == OVERLAY_TEXT.replace("0000000000000000", "abcdef0123456789")
    assert not pin(path, "abcdef0123456789")


def test_a_pin_reads_unquoted_too_and_is_written_quoted(tmp_path):
    path = tmp_path / "kustomization.yaml"
    path.write_text(OVERLAY_TEXT.replace('"0000000000000000"', "1234"))

    assert pinned_tag(path.read_text()) == "1234"
    assert pin(path, "1234")
    assert 'newTag: "1234"' in path.read_text()


def test_an_overlay_without_the_serving_image_is_refused(tmp_path):
    path = tmp_path / "kustomization.yaml"
    path.write_text("resources: []\n")

    with pytest.raises(ValueError, match="no newTag"):
        pin(path, "abc")


@needs_git
@pytest.mark.skipif(not OVERLAY.exists(), reason="no deploy/ (the image has none)")
def test_the_deployed_overlay_pins_the_image_built_from_this_commit():
    assert pinned_tag(OVERLAY.read_text()) == image_tag(inputs_listing(ROOT)), (
        "the kind overlay pins another image than this commit's; run "
        "`python3 scripts/serving/image_tag.py --pin deploy/k8s/overlays/kind/kustomization.yaml`"
        " (the pre-commit hook does) and commit the pin"
    )
