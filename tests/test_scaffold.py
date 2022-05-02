from pathlib import Path

import tomllib

import late_interaction


def test_scaffold_import_and_project_metadata() -> None:
    root = Path(__file__).parents[1]
    with (root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert late_interaction.__version__ == project["version"]
    assert project["name"] == "late-interaction-retrieval-pq"
    assert (root / "LICENSE").is_file()
    assert (root / ".github/workflows/ci.yml").is_file()
