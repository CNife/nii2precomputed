import os
import sys
from pathlib import Path

from loguru import logger
from typer import Argument, Option, Typer

from convert_to_precomputed.convert import LOG_FORMAT, image_2_precomputed
from convert_to_precomputed.io_utils import list_dir
from convert_to_precomputed.zimg_utils import read_image_info

URL: str = str(os.environ.get("URL", "http://10.11.40.170:2000"))
BASE_PATH: Path = Path(os.environ.get("BASE_PATH", "/zjbs-data/share"))

app = Typer(help="Convert images into Neuroglancer precomputed data")


@app.command(help="Convert multiple images on z dimension to precomputed")
def convert(
    image_path: Path = Argument(
        help="Image file or files directory", exists=True, dir_okay=True, file_okay=True, show_default=False
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: tuple[float, float, float] = Argument(help="resolution of x, y, z", min=0.0, default=(0.0, 0.0, 0.0)),
    z_range: tuple[int, int] = Option(help="Z range, -1 means end", default=(0, -1)),
    write_block_size: int = Option(help="Block size when writing precomputed", default=512),
    resume: bool = Option(help="Resume from output_directory/work_progress.json", default=True),
    base_url: str = Option(help="Base url in base.json", default="http://10.11.40.170:2000"),
    base_path: Path = Option(help="Base path, must be parent of output directory", default=Path("/zjbs-data/share")),
) -> None:
    logger.info(
        f"Converting image to precomputed: "
        f"image_path={str(image_path)},output_directory={str(output_directory)},{resolution=},{z_range=},"
        f"{write_block_size=},{resume=}"
    )
    if image_path.is_dir():
        image_path = list_dir(image_path)
    image_2_precomputed(
        image_path, output_directory, resolution, z_range, write_block_size, resume, base_url, base_path
    )


@app.command(help="Show single image or multiple images info")
def show_info(path: Path = Argument(exists=True, show_default=False)) -> None:
    if path.is_dir():
        image_info = read_image_info(list_dir(path))
    else:
        image_info = read_image_info(path)
    logger.info(f"path={str(path)}")
    logger.info(f"{image_info=}")


@app.command(help="Generate specification for image")
def gen_spec(
    image_path: Path = Argument(
        help="Image file or files directory", exists=True, dir_okay=True, file_okay=True, show_default=False
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: str = Argument(help="resolution of x, y, z", default="0.0,0.0,0.0"),
    write_block_size: int = Option(help="Block size when writing precomputed", default=512),
    base_url: str = Option(help="Base url in base.json", default="http://10.11.40.170:2000"),
    base_path: Path = Option(help="Base path, must be parent of output directory", default=Path("/zjbs-data/share")),
) -> None:
    pass


@app.command(help="Generate base.json for image")
def gen_base_json(
    image_path: Path = Argument(
        help="Image file or files directory", exists=True, dir_okay=True, file_okay=True, show_default=False
    ),
    output_directory: Path = Argument(help="Output directory", show_default=False),
    resolution: str = Argument(help="resolution of x, y, z", default="0.0,0.0,0.0"),
    write_block_size: int = Option(help="Block size when writing precomputed", default=512),
    base_url: str = Option(help="Base url in base.json", default="http://10.11.40.170:2000"),
    base_path: Path = Option(help="Base path, must be parent of output directory", default=Path("/zjbs-data/share")),
) -> None:
    pass


@app.command(help="Convert single scale for image")
def convert_scale(
    spec_path: Path = Argument(
        help="Specification for converting image", exists=True, file_okay=True, dir_okay=False, show_default=False
    ),
    scale_key: str = Argument(help="The scale to be converted in spec file", show_default=False),
) -> None:
    pass


@logger.catch
def main():
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT)
    app()


if __name__ == "__main__":
    main()
