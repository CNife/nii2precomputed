import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from zimg import ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate

from old_scripts.nii_2_precomputed.util import dbg_args


def convert_czi_2_single_image(image_path: str, out_dir: str, z: int) -> str:
    console = Console()

    result_file_name = f"luo_{z:04d}.nim"
    result_path = os.path.join(out_dir, result_file_name)
    if os.path.exists(result_path):
        console.log(f"SKIP {result_file_name}: file already exists")
        return result_path

    working_file = result_path + ".working"
    if os.path.exists(working_file):
        console.log(f"SKIP {result_file_name}: working file exists")
        return result_path
    with open(working_file, "w") as f:
        f.write(result_file_name)

    start_time = datetime.now()
    single_image_data = ZImg(
        image_path, region=ZImgRegion(ZVoxelCoordinate(0, 0, z, 0, 0), ZVoxelCoordinate(-1, -1, z + 1, -1, -1)), scene=0
    )
    single_image_data.save(result_path)
    end_time = datetime.now()
    used_seconds = math.ceil((end_time - start_time).total_seconds())

    os.remove(working_file)
    console.log(f"DONE [{end_time.strftime('%Y-%m-%d %H:%M:%S')}] {result_file_name} used {used_seconds}s")
    return result_path


def main(
    image_path: Path,
    out_dir: Path,
    start_z: int = 0,
    end_z: int = -1,
    reverse: bool = True,
    max_workers: int | None = None,
) -> None:
    if end_z < start_z:
        zimg_info: ZImgInfo = ZImg.readImgInfos(str(image_path))[0]
        end_z = zimg_info.depth
    z_range = range(start_z, end_z)
    dbg_args()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        result_files = executor.map(
            convert_czi_2_single_image,
            itertools.repeat(str(image_path)),
            itertools.repeat(str(out_dir)),
            reversed(z_range) if reverse else z_range,
        )
        result_files = list(result_files)
        print(f"{len(result_files)}/{end_z - start_z} images converted")


if __name__ == "__main__":
    typer.run(main)
