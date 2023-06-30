import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy import ndarray
from PIL import Image
from rich.console import Console
from rich.progress import Progress, track
from zimg import ZImg, ZImgInfo, ZImgRegion, ZImgSource, ZVoxelCoordinate

console = Console()


def read_image(path: Path, channel: int = 0) -> ZImg:
    region = ZImgRegion(
        ZVoxelCoordinate(x=0, y=0, z=0, c=channel, t=0),
        ZVoxelCoordinate(x=-1, y=-1, z=-1, c=channel + 1, t=1),
    )
    with Progress(console=console) as progress:
        progress.add_task("Reading image data", total=None)
        return ZImg(str(path), region)


def write_image(
    image: ndarray,
    z: int,
    output_dir: Path,
) -> int:
    image = normalize_higher_byte(image)
    image_path = output_dir / f"{z}.png"
    Image.fromarray(image, "L").save(image_path)
    return z


def new_main(image_file: Path, image_count: int, output_dir: Path) -> None:
    output_dir = output_dir / image_file.name
    output_dir.mkdir(parents=True, exist_ok=True)

    zimg_info = read_image_info(image_file)
    assert zimg_info.numChannels == 1
    sample_z = np.linspace(
        0, zimg_info.depth, image_count, endpoint=False, dtype=np.int32
    )
    console.print(f"Sample {image_count} images at {sample_z}")

    zimg = read_image(image_file)
    zimg_data = zimg.data[0][0]

    with ThreadPoolExecutor() as executor, Progress(console=console) as progress:
        pbar_task = progress.add_task(
            f"Writing {image_count} images", total=image_count
        )
        tasks = {
            z: executor.submit(write_image, zimg_data[z].copy(), z, output_dir)
            for z in sample_z
        }
        while tasks:
            finished = []
            for z, task in tasks.items():
                try:
                    finished_z = task.result(timeout=0.5)
                except TimeoutError:
                    continue
                else:
                    console.print(f"{finished_z}.png saved")
                    progress.advance(pbar_task)
                    finished.append(finished_z)
            for finished_z in finished:
                del tasks[finished_z]


def sample_images(
    data_path: Path, image_count: int, channel: int = 0
) -> Iterator[tuple[int, ndarray]]:
    data_info = read_image_info(data_path)
    sample_z = np.linspace(
        0, data_info.depth, image_count, endpoint=False, dtype=np.int32
    )
    for z in sample_z:
        yield z, read_single_image(data_path, z, channel)


def read_image_info(data_path: Path) -> ZImgInfo:
    # noinspection PyArgumentList
    info = ZImg.readImgInfo(ZImgSource(str(data_path)))
    console.print(info)
    return info


def read_single_image(data_path: Path, z: int, channel: int = 0) -> ndarray:
    image_region = ZImgRegion(
        ZVoxelCoordinate(x=0, y=0, z=z, c=channel, t=0),
        ZVoxelCoordinate(x=-1, y=-1, z=z + 1, c=channel + 1, t=1),
    )
    image = ZImg(str(data_path), image_region)
    # 复制数组，否则函数返回后ZImg会释放资源
    return image.data[0][0, 0].copy()


def normalize_higher_byte(array: ndarray) -> ndarray:
    assert array.dtype == np.uint16
    return np.right_shift(array, 8).astype(np.uint8)


def normalize_lower_byte(array: ndarray) -> ndarray:
    assert array.dtype == np.uint16
    return array.astype(np.uint8)


async def read_all_images(data_file: Path, output_images_dir: Path) -> None:
    output_dir = output_images_dir / data_file.name
    output_dir.mkdir(parents=True, exist_ok=True)

    zimg_obj = ZImg(str(data_file))
    zimg_data = zimg_obj.data[0][0]
    zimg_data = normalize_higher_byte(zimg_data)
    image_count = zimg_data.shape[0]

    with Progress() as progress:
        task = progress.add_task(f"Write {image_count} images", total=image_count)

        async def write_image(
            image: ndarray,
            z: int,
        ) -> None:
            image_path = output_dir / f"{z}.png"
            image = Image.fromarray(image, "L")
            image.save(image_path)
            progress.advance(task)

        await asyncio.gather(write_image(image, z) for z, image in enumerate(zimg_data))


def main(data_file: Path, image_count: int, output_images_dir: Path) -> None:
    output_dir = output_images_dir / data_file.name
    output_dir.mkdir(parents=True, exist_ok=True)

    for z, image in track(
        sample_images(data_file, image_count),
        description=f"Sample {image_count} images",
        total=image_count,
    ):
        image_path = output_dir / f"{z}.png"
        image = normalize_higher_byte(image)
        # image = normalize_lower_byte(image)
        Image.fromarray(image, "L").save(image_path)


if __name__ == "__main__":
    data_file = Path(r"C:\WorkData\nii\20230530\full16_100um_2009b_sym.nii.gz")
    output_dir = Path(__file__).parent / "output-images"
    # main(data_file, 10, output_dir)
    # asyncio.run(read_all_images(data_file, output_dir))
    new_main(data_file, 20, output_dir)
