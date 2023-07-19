from pathlib import Path

import typer

from convert_precomputed.rich_utils import print_args
from convert_precomputed.zimg_utils import read_image_info

app = typer.Typer(help="Convert images into Neuroglancer precomputed data")
app_options = {}


@app.command(help="Convert single image to precomputed")
def single_image(
    image_path: Path = typer.Argument(help="Image file path", exists=True),
    output_directory: Path = typer.Argument(help="Output directory"),
) -> None:
    pass


@app.command(help="Convert multiple images on z dimension to precomputed")
def multiple_images() -> None:
    pass


@app.command(help="Show single image or multiple images info")
def show_image_info(path: Path = typer.Argument(exists=True)) -> None:
    files = Path(path)
    if path.is_dir():
        files = list(path.iterdir())
        files.sort(key=str)
    image_info = read_image_info(files)
    print_args(path=str(path), image_info=str(image_info))


if __name__ == "__main__":
    app()
