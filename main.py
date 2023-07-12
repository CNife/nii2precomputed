from pathlib import Path

import typer

from nii_2_precomputed import Resolution, convert_nii_to_precomputed


def main(
    image_path: Path,
    resolution: int,
    base_dir: Path | None = None,
    base_url: str = "http://localhost:8080"
) -> None:
    if base_dir is None:
        base_dir = image_path.parent

    folder_name = image_path.name
    if (i := folder_name.find(".nii")) >= 0:
        folder_name = folder_name[:i]
    out_folder = image_path.parent / folder_name
    out_folder_url_path = "/".join(out_folder.relative_to(base_dir).parts)
    url_path = f"{base_url}/{out_folder_url_path}"
    resolution = Resolution(resolution, resolution, resolution)

    convert_nii_to_precomputed(out_folder, image_path, url_path, resolution)


if __name__ == '__main__':
    typer.run(main)
