import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import typer
from zimg import ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate

from util import dbg_args


def convert_czi_2_single_image(image_path: str, out_dir: str, z: int) -> str:
    start_time = datetime.now()
    single_image_data = ZImg(
        image_path,
        region=ZImgRegion(
            ZVoxelCoordinate(0, 0, z, 0, 0), ZVoxelCoordinate(-1, -1, z + 1, -1, -1)
        ),
        scene=0,
    )
    result_file_name = f"luo_{z:04d}.nim"
    result_path = os.path.join(out_dir, result_file_name)
    single_image_data.save(result_path)
    end_time = datetime.now()
    used_seconds = (end_time - start_time).total_seconds()
    print(
        f"DONE [{end_time.strftime('%Y-%m-%d %H:%M:%S')}] used {used_seconds}s : {result_file_name}"
    )
    return result_path


def main(
    image_path: Path,
    out_dir: Path,
    start_z: int = 0,
    end_z: int = -1,
    reverse: bool = True,
    max_workers: int | None = None
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
