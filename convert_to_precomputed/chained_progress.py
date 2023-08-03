import json
from reprlib import recursive_repr
from typing import Callable, Iterator, Optional, Sequence, TypeVar

from convert_to_precomputed.io_utils import dump_json
from convert_to_precomputed.types import JsonObject, OsPath

T = TypeVar("T")


class ChainedProgress:
    def __init__(self, name: str, parent: Optional["ChainedProgress"]):
        self.name: str = name
        self.parent: Optional[ChainedProgress] = parent
        self.children: list[ChainedProgress] = []
        self.index: int = 0
        self.count: int = 1
        self.description: str = ""

    def get_or_add(self, name: str) -> "ChainedProgress":
        for child in self.children:
            if child.name == name:
                return child
        child = ChainedProgress(name, self)
        self.children.append(child)
        return child

    def bind(self, seq: Sequence[T], gen_desc: Callable[[T], str] | None = None) -> "ChainedProgressIter[T]":
        return ChainedProgressIter(self, seq, gen_desc)

    def __str__(self) -> str:
        parent = f"{self.parent} " if self.parent else ""
        description = f" {self.description}" if self.description else ""
        return f"{parent}[{self.name} {self.index}/{self.count}{description}]"

    @recursive_repr()
    def __repr__(self) -> str:
        return (
            f"ChainedProgressNode(name={self.name},index={self.index},count={self.count},"
            f"description={self.description!r},parent={self.parent},children={self.children})"
        )

    def save(self, path: OsPath, backtrack_to_root: bool = True, step_back: bool = True) -> None:
        def serial_to_dict(progress: ChainedProgress):
            return {
                "name": progress.name,
                "index": progress.index - 1 if step_back else progress.index,
                "count": progress.count,
                "description": progress.description,
                "children": [serial_to_dict(child) for child in progress.children],
            }

        root = self
        while backtrack_to_root and root.parent:
            root = root.parent
        json_dict = serial_to_dict(root)
        dump_json(json_dict, path)

    @staticmethod
    def load(path: OsPath) -> "ChainedProgress":
        def load_from_dict(d: JsonObject, parent: ChainedProgress | None) -> ChainedProgress:
            progress = ChainedProgress(d["name"], None)
            progress.index = d["index"]
            progress.count = d["count"]
            progress.description = d["description"]
            progress.parent = parent
            progress.children = [load_from_dict(child_dict, progress) for child_dict in d["children"]]
            return progress

        with open(path, "r") as f:
            json_dict = json.load(f)
            return load_from_dict(json_dict, None)


class ChainedProgressIter(Iterator[T]):
    def __init__(self, progress: ChainedProgress, seq: Sequence[T], gen_desc: Callable[[T], str] | None):
        self.progress: ChainedProgress = progress
        self.seq: Sequence[T] = seq
        self.gen_desc: Callable[[T], str] = gen_desc
        self.is_first: bool = True

        self.progress.count = len(self.seq)

    def __next__(self) -> T:
        if self.progress.index >= self.progress.count:
            raise StopIteration
        next_item = self.seq[self.progress.index]
        if self.is_first:
            self.is_first = False
        else:
            self.progress.children.clear()
        self.progress.index += 1
        if self.gen_desc:
            self.progress.description = self.gen_desc(next_item)
        return next_item
