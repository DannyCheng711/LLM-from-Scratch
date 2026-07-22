import csv
import json
import time
from pathlib import Path

class ExperimentLogger:
    """
    Automatically creates

    log_dir/
        config.json
        metrics.csv
        train.log
    """

    def __init__(self, log_dir, config=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.log_dir / "metrics.csv"
        self.log_path = self.log_dir / "train.log"
        self.config_path = self.log_dir / "config.json"

        self.start_time = time.time()

        # Save hyperparameters once
        if config is not None:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent = 4)

        # Write CSV header
        if not self.metrics_path.exists():
            with open(self.metrics_path, "w", newline = "") as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        "step",
                        "wallclock_seconds",
                        "train_loss",
                        "val_loss",
                        "learning_rate",
                    ]
                )


    def log(self, step, train_loss=None, val_loss=None, learning_rate=None):
        """
        log one training/evaluation record
        """
        elapsed = time.time() - self.start_time

        with open(self.metrics_path, "a", newline = "") as f:

            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    round(elapsed, 2),
                    train_loss,
                    val_loss,
                    learning_rate,
                ]
            )

        message = (
            f"Step {step:>6} | "
            f"Time {elapsed:8.2f}s | "
            f"Train {train_loss} | "
            f"Val {val_loss} | "
            f"LR {learning_rate}"
        )

        with open(self.log_path, "a") as f:
            f.write(message + "\n")

        print(message)