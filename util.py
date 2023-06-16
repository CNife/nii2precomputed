import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

console = Console()


def pretty_dump_json(o: Any) -> str:
    return json.dumps(o, indent=2, sort_keys=True)


def pretty_print_object(o: Any, title: str | None = None) -> None:
    panel = Panel(Pretty(o), title=title)
    console.print(panel)
