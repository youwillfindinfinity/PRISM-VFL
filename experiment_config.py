"""Single source of truth for experiment settings; run-time values are logged by train.run_training."""

from pathlib import Path

SEEDS          = [42, 123, 7]                       # manuscript experiments
SWEEP_SEEDS    = [7, 42, 123, 0, 1, 2, 3, 4, 5, 6]  # revision privacy sweep (10 seeds)

BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
MAX_ROUNDS     = 100
PATIENCE       = 15                                 # rounds without improvement (§2.4.3)
TASK_WEIGHTS   = {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0}   # equal weights (Eq. 3)
CHECKPOINT_SELECTION = "best mean validation AUROC over active tasks"

# Differential privacy (cut layer)
CLIPPING_NORM  = 1.0
DELTA          = 1e-5
EPSILON_LEVELS = [float("inf"), 10.0, 5.0, 2.0, 1.0, 0.5]


_GIT = Path(__file__).resolve().parent / ".git"


def git_branch() -> str:
    """Checked-out branch read from .git/HEAD, without running git."""
    try:
        head = (_GIT / "HEAD").read_text().strip()
    except OSError:
        return "unknown"
    return head.removeprefix("ref: refs/heads/") if head.startswith("ref: ") else "detached"


def git_commit() -> str:
    """HEAD commit read from .git files, without running git."""
    git = _GIT
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: "):]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"
