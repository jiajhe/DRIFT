"""Run the revised DRIFT training recipe and its locked test comparisons on a single GPU."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/drift-tslm-v2.1"
ENTRY = ROOT / "code/TSLM_training/run_v2.sh"


def run_phase(variant, phase):
    path = RUN / f"{variant}-{phase}.log"
    for micro in (16, 8, 4, 2, 1):
        with path.open("a+") as log:
            start = log.tell()
            result = subprocess.run(
                [
                    "bash",
                    str(ENTRY),
                    "--variant",
                    variant,
                    "--phase",
                    phase,
                    "--microbatch",
                    str(micro),
                    "--eval-batch",
                    str(4 * micro),
                ],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            if result.returncode == 0:
                return
            log.seek(start)
            if "out of memory" not in log.read().lower():
                raise SystemExit(
                    f"{variant} {phase} exited {result.returncode}. See {path}"
                )
        print(f"{variant} {phase} exhausted memory at batch {micro}", flush=True)
    raise SystemExit("Insufficient GPU memory at all supported microbatches")


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    for variant, phase in (
        ("drift", "train"),
        ("drift", "test"),
    ):
        run_phase(variant, phase)
        print(variant, phase, "finished", flush=True)
    subprocess.run(
        [sys.executable, str(ENTRY.with_name("v2_report.py"))], cwd=ROOT, check=True
    )
