import os
from pathlib import Path

URL: str = str(os.environ.get("URL", "http://10.11.40.170:2000"))
BASE_PATH: Path = Path(os.environ.get("BASE_PATH", "/zjbs-data/share"))
