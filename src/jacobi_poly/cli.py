import typer

app = typer.Typer()


@app.command()
def benchmark() -> None:
    """Benchmark the Jacobi polynomial evaluation."""
