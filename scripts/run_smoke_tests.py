#!/usr/bin/env python3
from __future__ import annotations

"""Repository smoke tests that do not require the research dependencies."""

import argparse
import compileall
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SCRIPTS_DIR = REPO_ROOT / "scripts"


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Run static and CLI smoke tests for the repository.")


def validate_configs() -> list[Path]:
    validated: list[Path] = []
    for config_path in sorted(CONFIGS_DIR.glob("*.json")):
        with config_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        validated.append(config_path)
    return validated


def validate_notebooks() -> list[Path]:
    validated: list[Path] = []
    for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        with notebook_path.open("r", encoding="utf-8") as handle:
            notebook = json.load(handle)
        cells = notebook.get("cells", [])
        if not cells or cells[0].get("cell_type") != "markdown":
            raise RuntimeError(f"{notebook_path.name} is missing the required leading markdown note.")
        validated.append(notebook_path)
    return validated


def validate_script_help() -> list[Path]:
    validated: list[Path] = []
    for script_path in sorted(SCRIPTS_DIR.glob("run_*.py")):
        subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        validated.append(script_path)
    return validated


def main() -> None:
    build_parser().parse_args()

    if not compileall.compile_dir(REPO_ROOT / "src", quiet=1):
        raise SystemExit("compileall failed for src/")
    if not compileall.compile_dir(REPO_ROOT / "scripts", quiet=1):
        raise SystemExit("compileall failed for scripts/")

    configs = validate_configs()
    notebooks = validate_notebooks()
    scripts = validate_script_help()

    print(f"Compiled src/ and scripts/ successfully.")
    print(f"Validated {len(configs)} JSON configs.")
    print(f"Validated {len(notebooks)} notebooks.")
    print(f"Validated {len(scripts)} CLI help commands.")


if __name__ == "__main__":
    main()
