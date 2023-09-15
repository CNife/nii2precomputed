import os.path
import subprocess
import sys
import sysconfig

src_zimg_jars_dir = os.path.join(sysconfig.get_path("platlib"), "zimg", "jars")

subprocess.run(
    [
        sys.executable,
        "-m",
        "nuitka",
        "--standalone",
        f"--include-data-dir={src_zimg_jars_dir}=zimg/jars",
        f"--output-filename=convert_to_precomputed",
        "main.py",
    ],
    check=True,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
