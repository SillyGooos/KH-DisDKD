import csv
from pathlib import Path


class LossTracker:
    def __init__(self, log_path: str, method: str):
        self.log_path = Path(log_path)
        self.method = method

        self.headers = ["epoch", "phase", "total", "ce", "kd", "accuracy"]

        method_headers = {
            "FitNet": ["hint"],
            "CRD": ["contrastive"],
            "DKD": ["tckd", "nckd"],
            "DisDKD": [
                "dkd",
                "discriminator",
                "adversarial",
                "disc_accuracy",
                "fool_rate",
            ],
        }
        if method in method_headers:
            self.headers.extend(method_headers[method])

        # Optional but highly recommended:
        self.headers.append("lr")

        self._write_headers()

    def _write_headers(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.headers)

    @staticmethod
    def _to_float(x):
        try:
            # handles torch scalars too
            return float(x)
        except Exception:
            return 0.0

    def log_epoch(
        self, epoch: int, phase: str, losses: dict, accuracy: float, lr: float = 0.0
    ):
        row = [
            epoch,
            phase,
            self._to_float(losses.get("total", 0.0)),
            self._to_float(losses.get("ce", 0.0)),
            self._to_float(losses.get("kd", 0.0)),
            self._to_float(accuracy),
        ]

        method_losses = {
            "FitNet": ["hint"],
            "CRD": ["contrastive"],
            "DKD": ["tckd", "nckd"],
            "DisDKD": [
                "dkd",
                "discriminator",
                "adversarial",
                "disc_accuracy",
                "fool_rate",
            ],
        }
        for k in method_losses.get(self.method, []):
            row.append(self._to_float(losses.get(k, 0.0)))

        row.append(self._to_float(lr))

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
