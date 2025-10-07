from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import time

console = Console()

def loading_animation(task_desc: str, func, *args, **kwargs):
    """Show a spinner while running a blocking task."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(task_desc, start=False)
        progress.start_task(task)
        result = func(*args, **kwargs)
        time.sleep(0.3)  # small pause for smoothness
        return result
