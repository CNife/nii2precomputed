"""沿Z轴方向拼接多个NII文件片段
步骤：
1. 分别读取多个NII文件的元信息，确认X，Y的尺寸相同，计算各自的Z轴位置
2. 根据X，Y，以及拼接后的Z轴尺寸，计算多分辨率缩放信息和base.json
3. 分别写入多分辨率缩放尺寸、多文件的precomputed数据
"""
from pathlib import Path

import typer

from util import pretty_print_object


def main(
    image_files_dir: Path,
    resolution: int,
    base_dir: Path = Path(r"C:\Workspace"),
    url_base: str = "http://localhost:8080",
) -> None:
    pretty_print_object(
        {
            "image_files_dir": str(image_files_dir.absolute()),
            "resolution": resolution,
            "base_dir": str(base_dir),
            "url_base": url_base,
        },
        "Arguments",
    )


if __name__ == "__main__":
    typer.run(main)
