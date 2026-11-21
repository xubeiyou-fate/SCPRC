"""Resolve current and archived artifact pointers within this checkout only.

Archived source paths are lookup keys, never executable fallback locations.
"""
from functools import lru_cache
from pathlib import Path
import csv
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ANCHORS = {"scprc", "experiments", "results", "checkpoints", "configs",
            "manifests", "docs", "reproduction", "research", "paper2"}


def _suffix(value: str | Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        if path.is_relative_to(PROJECT_ROOT):
            return path.relative_to(PROJECT_ROOT).as_posix()
        for index, part in enumerate(path.parts):
            if part in _ANCHORS:
                return Path(*path.parts[index:]).as_posix()
        raise ValueError(f"Artifact path has no project-relative suffix: {path}")
    return path.as_posix()


@lru_cache(maxsize=1)
def _artifact_aliases():
    aliases = {}
    directories = {}

    def remember(source, destination):
        aliases[str(Path(source).expanduser())] = destination
        try:
            aliases[_suffix(source)] = destination
        except ValueError:
            # A manifest may identify an exported artifact without a project
            # suffix. Only its exact archived path is an accepted lookup key.
            pass

    naming = PROJECT_ROOT / "manifests/NAMING_AUDIT.json"
    if naming.is_file():
        for row in json.loads(naming.read_text())["paths"]:
            old, new = row["source_path"], row["published_path"]
            aliases[old] = new
            if not old.startswith("checkpoints/"):
                left, right = Path(old).parent, Path(new).parent
                directories.setdefault(left.as_posix(), set()).add(right.as_posix())
                while left.name == right.name and left.name:
                    left, right = left.parent, right.parent
                    directories.setdefault(left.as_posix(), set()).add(right.as_posix())
    for filename, records, target in (
        ("CODE_MIGRATION_AUDIT.json", "copied_files", "path"),
        ("EVIDENCE_MIGRATION_AUDIT.json", "files", "destination"),
    ):
        manifest = PROJECT_ROOT / "manifests" / filename
        if manifest.is_file():
            for row in json.loads(manifest.read_text()).get(records, []):
                if row.get("source_path") and row.get(target):
                    remember(row["source_path"], row[target])
    checkpoints = PROJECT_ROOT / "manifests/CHECKPOINTS.csv"
    if checkpoints.is_file():
        with checkpoints.open(newline="") as stream:
            for row in csv.DictReader(stream):
                for key in ("original_source", "resolved_source"):
                    if row.get(key):
                        remember(row[key], row["repo_relative"])
                        try:
                            old = _suffix(row[key])
                        except ValueError:
                            continue
                        aliases["checkpoints/" + old] = row["repo_relative"]
    prefix = {old: next(iter(new)) for old, new in directories.items()
              if len(new) == 1 and old not in {".", ""}}
    prefix.update({"reproduction/optional_baselines": "scprc/baselines",
                   "reproduction": "scprc",
                   "research/urbanev_timemixer/experiments": "scprc/training",
                   "research/urbanev_timemixer/external_data": "scprc/data_adapters",
                   "paper2/scripts": "experiments/commands"})
    return aliases, sorted(prefix.items(), key=lambda item: -len(item[0]))


def project_artifact_path(value: str | Path) -> Path:
    aliases, prefixes = _artifact_aliases()
    exact = str(Path(value).expanduser())
    relative = aliases[exact] if exact in aliases else _suffix(value)
    if relative in aliases:
        relative = aliases[relative]
    else:
        for old, new in prefixes:
            if relative == old or relative.startswith(old + "/"):
                relative = new + relative[len(old):]
                break
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Artifact path escapes this checkout: {value}")
    return path
