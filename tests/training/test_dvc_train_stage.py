"""The DVC `train`, `evaluate` and `gate` stages' code deps must cover
everything their entry points import.

They're listed module by module (not the whole package) so unrelated edits
don't force a re-run; a module an entry point starts importing without being
added to dvc.yaml would let `dvc repro` call a stale model or stale scores up
to date. These tests compute each entry point's transitive imports of this
package and fail if any of their files isn't covered by a dep.

The model registry imports models lazily, by import path, so the import
graph doesn't reach the model the stages train and score: the closure adds
that model's module, looked up from the `train` stage's `model=` override.
Likewise `run.py` imports the final scoring (`training/scoring.py`) by name,
only when a run scores itself; the `train` stage doesn't
(`evaluation.final=false`), which is what keeps the evaluation-only code out
of its deps -- checked here too.
"""

import ast
from pathlib import Path

import pytest
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


TRAIN = ROOT / "scripts" / "training" / "train.py"
EVALUATE = ROOT / "scripts" / "evaluation" / "evaluate_checkpoint.py"
GATE = ROOT / "scripts" / "evaluation" / "gate_checkpoint.py"


def stage(name: str) -> dict:
    return yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"][name]


def train_stage() -> dict:
    return stage("train")


def trained_model_module(cmd: str) -> str:
    """The module of the model a train command builds: its `model=` override
    names a `configs/model/` file, whose `name` is a registry entry."""
    (choice,) = [arg.partition("=")[2] for arg in cmd.split() if arg.startswith("model=")]
    config = yaml.safe_load((ROOT / "configs" / "model" / f"{choice}.yaml").read_text())
    return MODELS[config["name"]].partition(":")[0]


def import_closure(entry_point: Path) -> set[Path]:
    """The files of every module of this package `entry_point` imports,
    transitively, plus the trained model's."""
    files: set[Path] = set()
    todo = list(imported_modules(entry_point))
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


def train_import_closure() -> set[Path]:
    return import_closure(TRAIN)


@pytest.mark.parametrize(
    "name, entry_point", [("train", TRAIN), ("evaluate", EVALUATE), ("gate", GATE)]
)
def test_stage_deps_cover_everything_its_entry_point_imports(name, entry_point):
    deps = [ROOT / dep for dep in stage(name)["deps"]]

    assert entry_point in deps
    missing = sorted(
        str(p.relative_to(ROOT)) for p in import_closure(entry_point) if not covered(p, deps)
    )
    assert not missing, f"add to the {name} stage's deps in dvc.yaml: {missing}"


def test_the_train_stage_leaves_scoring_to_the_evaluate_stage():
    """The train stage doesn't score its model, so the by-name import of the
    final scoring never runs in it and the evaluation-only code (diagnostics,
    ensemble scores) isn't among what it depends on."""
    train, evaluate = train_stage(), stage("evaluate")
    closure = train_import_closure()

    assert "evaluation.final=false" in train["cmd"].split()
    assert ROOT / "src" / PACKAGE / "training" / "scoring.py" not in closure
    assert ROOT / "src" / PACKAGE / "evaluation" / "diagnostics.py" not in closure
    assert ROOT / "src" / PACKAGE / "evaluation" / "selection.py" in closure
    # evaluate scores what train stored, and writes the metrics.
    (checkpoint,) = train["outs"]
    assert checkpoint in evaluate["deps"]
    assert f"checkpoint={checkpoint}" in evaluate["cmd"].split()
    assert "metrics" in evaluate and "metrics" not in train


def test_the_model_stages_use_deterministic_kernels():
    for name in ("train", "evaluate", "gate"):
        assert "jax.deterministic_ops=true" in stage(name)["cmd"].split()


def test_the_gate_checks_what_train_stored_against_what_evaluate_scored():
    train, evaluate, gate = stage("train"), stage("evaluate"), stage("gate")
    (checkpoint,) = train["outs"]
    ((metrics, _),) = (item for entry in evaluate["metrics"] for item in entry.items())

    assert checkpoint in gate["deps"] and metrics in gate["deps"]
    assert f"checkpoint={checkpoint}" in gate["cmd"].split()
    assert f"export.dir={Path(metrics).parent}" in gate["cmd"].split()


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
