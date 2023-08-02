import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Callable, Iterable, Iterator, Optional, T

from loguru import logger

from convert_precomputed.types import JsonObject, T

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|"
    "<level>{level: <8}</level>|"
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>|"
    "<level>{message}</level>"
)


@contextmanager
def log_time_usage(description: str) -> Iterable[None]:
    start_time = time.perf_counter_ns()
    try:
        yield
    finally:
        time_diff_ns = time.perf_counter_ns() - start_time
        used_time = timedelta(microseconds=time_diff_ns / 1000)
        logger.info(f"[used {str(used_time)}] {description}")


class ChainedIndexProgress:
    def __init__(
        self,
        parent: Optional["ChainedIndexProgress"],
        name: str,
        description: str = "",
        index: int = 0,
        count: int = 1,
        children: list["ChainedIndexProgress"] = None,
    ):
        self.parent: Optional["ChainedIndexProgress"] = parent
        self.name: str = name
        self.description: str = description
        self.children: list["ChainedIndexProgress"] = [] if children is None else children
        self.index: int = index
        self.count: int = count

        if self.parent is not None:
            # noinspection PyUnresolvedReferences
            self.parent.children.append(self)

    def __str__(self):
        description_str = f" {self.description}" if self.description else ""
        self_str = f"[{self.name} {self.index}/{self.count}{description_str}]"
        if self.parent is None:
            return self_str
        return f"{str(self.parent)} {self_str}"

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "description": self.description,
            "index": self.index - 1,
            "count": self.count,
            "children": [child.to_dict() for child in self.children],
        }

    @staticmethod
    def from_dict(d: JsonObject, parent: Optional["ChainedIndexProgress"] = None) -> "ChainedIndexProgress":
        obj = ChainedIndexProgress(
            parent, name=d["name"], description=d["description"], index=d["index"], count=d["count"]
        )
        for child_dict in d["children"]:
            ChainedIndexProgress.from_dict(child_dict, obj)
        return obj

    def bind_list(self, iterable: list[T], gen_desc: Callable[[T], str] | None = None) -> "ChainedIndexProgressIter":
        return ChainedIndexProgressIter(iterable, self, gen_desc)


class ChainedIndexProgressIter(Iterator[T]):
    def __init__(self, list_data: list[T], progress: ChainedIndexProgress, gen_desc: Callable[[T], str] | None):
        self.list_data: list[T] = list_data
        self.progress: ChainedIndexProgress = progress
        self.gen_desc: Callable[[T], str] | None = gen_desc
        self.started = False

        self.progress.count = len(self.list_data)
        if self.gen_desc is not None:
            self.progress.description = self.gen_desc(self.list_data[0])

    def __iter__(self):
        return self

    def __next__(self) -> T:
        if self.progress.index >= len(self.list_data):
            raise StopIteration()
        item = self.list_data[self.progress.index]
        self.progress.index += 1
        if self.started:
            for child in self.progress.children:
                child.index = 0
        else:
            self.started = True
        if self.gen_desc is not None:
            self.progress.description = self.gen_desc(item)
        return item
