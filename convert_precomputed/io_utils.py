import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from convert_precomputed.config import DEFAULT_BASE_PATH
from convert_precomputed.types import Json, OsPath


def dump_json(obj: Json, path: OsPath) -> None:
    with open(path, "w") as json_file:
        json.dump(obj, json_file, indent=2)


def list_dir(path: Path) -> list[Path]:
    files = [item for item in path.iterdir() if item.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def check_output_directory(path: Path, base_path: Path = DEFAULT_BASE_PATH) -> str:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise ValueError(f"{path} is not directory")
    try:
        return str(path.relative_to(base_path))
    except ValueError as e:
        raise ValueError(f"{path} is not subdirectory of {base_path}") from e


@contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(_to):
        sys.stdout.close()
        os.dup2(_to.fileno(), fd)
        sys.stdout = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(file)
        try:
            yield
        finally:
            _redirect_stdout(old_stdout)
