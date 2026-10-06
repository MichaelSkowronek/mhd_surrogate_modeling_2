"""The DVC `train` stage's code deps must cover everything train.py imports.

They're listed module by module (not the whole package) so unrelated edits
don't force a retrain; a module train.py starts importing without being
added to dvc.yaml would let `dvc repro` call a stale model up to date. This
test computes train.py's transitive imports of this package and fails if any
of their files isn't covered by a dep.

The model registry imports models lazily, by import path, so the import
graph doesn't reach the model the stage trains: the closure adds that
model's module, looked up from the stage's `model=` override.
"""

import ast
from pathlib import Path

import yaml

from mhd_surrogate.models.registry import MODELS

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "mhd_surrogate"


def imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(PACKAGE):
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(a.name for a in node.names if a.name.startswith(PACKAGE))
    return modules


def module_files(module: str) -> list[Path]:
    """The module's file plus the __init__.py of every package above it,
    all of which run when it's imported."""
    parts = module.split(".")
    files = [ROOT / "src" / Path(*parts[:i]) / "__init__.py" for i in range(1, len(parts))]
    as_file = ROOT / "src" / Path(*parts).with_suffix(".py")
    files.append(as_file if as_file.exists() else ROOT / "src" / Path(*parts) / "__init__.py")
    return files


def train_stage() -> dict:
    return yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]["train"]


def trained_model_module(cmd: str) -> str:
    """The module of the model a train command builds: its `model=` override
    names a `configs/model/` file, whose `name` is a registry entry."""
    (choice,) = [arg.partition("=")[2] for arg in cmd.split() if arg.startswith("model=")]
    config = yaml.safe_load((ROOT / "configs" / "model" / f"{choice}.yaml").read_text())
    return MODELS[config["name"]].partition(":")[0]


def train_import_closure() -> set[Path]:
    files: set[Path] = set()
    todo = list(imported_modules(ROOT / "scripts" / "training" / "train.py"))
    todo.append(trained_model_module(train_stage()["cmd"]))
    seen: set[str] = set()
    while todo:
        module = todo.pop()
        if module in seen:
            continue
        seen.add(module)
        for path in module_files(module):
            files.add(path)
            todo.extend(imported_modules(path))
    return files


def covered(path: Path, deps: list[Path]) -> bool:
    return any(path == dep or dep in path.parents for dep in deps)


def test_train_stage_deps_cover_everything_train_py_imports():
    deps = [ROOT / dep for dep in train_stage()["deps"]]

    assert ROOT / "scripts" / "training" / "train.py" in deps
    missing = sorted(
        str(p.relative_to(ROOT)) for p in train_import_closure() if not covered(p, deps)
    )
    assert not missing, f"add to the train stage's deps in dvc.yaml: {missing}"


def test_closure_finds_transitive_imports():
    """Sanity check of the helper: the trained model (Hankel DMD, reached only
    through the registry) imports the DMD model, which train.py never names;
    models nothing imports stay out."""
    closure = train_import_closure()

    assert ROOT / "src" / PACKAGE / "models" / "dmd.py" in closure
    assert ROOT / "src" / PACKAGE / "analysis" / "fields.py" in closure
    assert ROOT / "src" / PACKAGE / "models" / "baselines.py" not in closure


def test_trained_model_module_follows_the_model_override():
    cmd = "uv run scripts/training/train.py model=persistence export.dir=x"

    assert trained_model_module(cmd) == "mhd_surrogate.models.baselines"
