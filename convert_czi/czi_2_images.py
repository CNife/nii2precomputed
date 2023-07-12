import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import typer
from zimg import ZImg, ZImgInfo, ZImgRegion, ZVoxelCoordinate


def convert_czi_2_single_image(image_path: str, out_dir: str, z: int) -> str:
    single_image_data = ZImg(
        image_path,
        region=ZImgRegion(
            ZVoxelCoordinate(0, 0, z, 0, 0), ZVoxelCoordinate(-1, -1, z + 1, -1, -1)
        ),
        scene=0,
    )
    result_path = os.path.join(out_dir, f"luo_{z:04d}.nim")
    single_image_data.save(result_path)
    print(f"DONE: {result_path}")
    return result_path


def main(
    image_path: Path,
    out_dir: Path,
    start_z: int = 0,
    end_z: int = -1,
) -> None:
    if end_z < start_z:
        # noinspection PyTypeChecker
        zimg_info: ZImgInfo = ZImg.readImgInfos(str(image_path))[0]
        end_z = zimg_info.depth
    with ProcessPoolExecutor() as executor:
        result_files = executor.map(
            convert_czi_2_single_image,
            itertools.repeat(str(image_path)),
            itertools.repeat(str(out_dir)),
            range(start_z, end_z),
        )
        result_files = list(result_files)
        print(f"{len(result_files)}/{end_z - start_z} images converted")


if __name__ == "__main__":
    typer.run(main)
