import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Iterable, Optional

from loguru import logger


@contextmanager
def log_time_usage(description: str) -> Iterable[None]:
    start_time = time.perf_counter_ns()
    try:
        yield
    finally:
        time_diff_ns = time.perf_counter_ns() - start_time
        used_time = timedelta(microseconds=time_diff_ns / 1000)
        logger.info(f"{description} used {str(used_time)}")


class ChainedIndexProgress:
    def __init__(self, parent: Optional['ChainedIndexProgress'], name: str, description: str = "", count: int = 1):
        self.parent: Optional['ChainedIndexProgress'] = parent
        self.children: list['ChainedIndexProgress'] = []
        self.name: str = name
        self.description: str = description
        self.index: int = 0
        self.count: int = count

        if self.parent is not None:
            self.parent.children.append(self)

    def next(self):
        self.index += 1
        for child in self.children:
            child.index = 0

    def __str__(self):
        description_str = f" {self.description}" if self.description else ''
        self_str = f"[{self.name} {self.index}/{self.count}{description_str}]"
        if self.parent is None:
            return self_str
        return f"{str(self.parent)} {self_str}"
