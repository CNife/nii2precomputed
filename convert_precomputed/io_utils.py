import json

from convert_precomputed.types import Json, OsPath


def dump_json(obj: Json, path: OsPath) -> None:
    with open(path, "w") as json_file:
        json.dump(obj, json_file, indent=2)
