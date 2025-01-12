# main.py
import typer
from src.train import train_model
from src.evaluate import evaluate_model
import os

app = typer.Typer(add_completion=True, pretty_exceptions_show_locals=False, pretty_exceptions_enable=True)

@app.command()
def train(resume: bool = typer.Option(False, help="Resume training if possible.")):
    run_id = None
    experiments_dir = "experiments"
    run_id_file = os.path.join(experiments_dir, "run_id.txt")

    # Check if run_id_file exists
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = int(f.read().strip())
    else:
        run_id = int(resume)

    if not resume:
        # Ensure the experiments directory exists
        os.makedirs(experiments_dir, exist_ok=True)

        run_id += 1

        # Write the new run_id back to the file
        with open(run_id_file, "w") as f:
            f.write(str(run_id))

    train_model(run_id=run_id, resume=resume)

@app.command()
def evaluate(model: str = typer.Argument(..., help="Path to the saved model")):
    evaluate_model(model)

if __name__ == "__main__":
    app()