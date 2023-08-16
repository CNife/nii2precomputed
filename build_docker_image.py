import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from typer import Argument, Option


def main(
    repo: str = Argument("cnife/convert_to_precomputed"),
    additional_tags: Optional[list[str]] = Argument(None),
    push: bool = True,
    dry_run: bool = Option(False, is_flag=True),
) -> None:
    tags = ["latest", datetime.now().strftime("%Y%m%d-%H%M%S")]
    if additional_tags:
        tags.extend(additional_tags)
    build_cmd = ["docker", "build"]
    for tag in tags:
        build_cmd.append("--tag")
        build_cmd.append(f"{repo}:{tag}")
    if push:
        build_cmd.append("--push")
    build_cmd.append(str(Path(__file__).parent))
    run(build_cmd, dry_run)


def run(cmd: list[str], dry_run: bool):
    logger.info(" ".join(cmd))
    if not dry_run:
        return subprocess.run(cmd, check=True)


if __name__ == "__main__":
    typer.run(main)
