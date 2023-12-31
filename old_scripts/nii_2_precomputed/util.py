import inspect
import json
from typing import Any, Iterable, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

console = Console()
T = TypeVar("T")


def pretty_dump_json(o: Any) -> str:
    return json.dumps(o, indent=2, sort_keys=True)


def dbg(o: Any, title: str | None = None, subtitle: str | None = None) -> None:
    panel = Panel(Pretty(o, indent_size=2), title=title, subtitle=subtitle)
    console.print(panel)


def dbg_args() -> None:
    caller_frame = inspect.currentframe().f_back
    caller_locals = caller_frame.f_locals
    caller_func_name = caller_frame.f_code.co_name
    caller_file = caller_frame.f_code.co_filename
    caller_lineno = caller_frame.f_lineno
    dbg(
        caller_locals,
        f"Arguments of [bright_yellow]{caller_func_name}",
        f"{caller_file}:{caller_lineno}",
    )


def humanize_size(size: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}PB"


def chunks(lst: list[T], chunk_size: int) -> Iterable[list[T]]:
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def ranges(start: int, end: int, step: int) -> Iterable[tuple[int, int]]:
    for i in range(start, end, step):
        yield i, min(i + step, end)
