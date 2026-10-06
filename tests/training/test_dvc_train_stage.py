"""The DVC `train` stage's code deps must cover everything train.py imports.

They're listed module by module (not the whole package) so unrelated edits
don't force a retrain; a module train.py starts importing without being
added to dvc.yaml would let `dvc repro` call a stale model up to date. This
test computes train.py's transitive imports of this package and fails if any
of their files isn't covered by a dep.
"""

import ast
from pathlib import Path

import yaml

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


def train_import_closure() -> set[Path]:
    files: set[Path] = set()
    todo = list(imported_modules(ROOT / "scripts" / "training" / "train.py"))
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
    stage = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]["train"]
    deps = [ROOT / dep for dep in stage["deps"]]

    assert ROOT / "scripts" / "training" / "train.py" in deps
    missing = sorted(
        str(p.relative_to(ROOT)) for p in train_import_closure() if not covered(p, deps)
    )
    assert not missing, f"add to the train stage's deps in dvc.yaml: {missing}"


def test_closure_finds_transitive_imports():
    """Sanity check of the helper: the registry is imported by train.py and
    imports the DMD model, which train.py never names."""
    closure = train_import_closure()

    assert ROOT / "src" / PACKAGE / "models" / "dmd.py" in closure
    assert ROOT / "src" / PACKAGE / "analysis" / "fields.py" in closure
