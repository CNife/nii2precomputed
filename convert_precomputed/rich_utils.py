from rich.console import Console
from rich.table import Table

console = Console()


def print_args(**kwargs) -> None:
    grid = Table.grid(expand=True)
    grid.add_column("key", justify="right")
    grid.add_column(":", justify="left")
    grid.add_column("value", justify="left", overflow="fold")
    for key, value in kwargs.items():
        grid.add_row(key, ": ", value)
    console.print(grid)
