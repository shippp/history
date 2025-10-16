import contextlib
import sys
from pathlib import Path


@contextlib.contextmanager
def live_log_redirect(log_file: str | Path):
    """
    Context manager that redirects all print() output (stdout + stderr)
    to a log file in real time (flushed at every line).

    Args:
        log_file (str | Path): Path to the log file.

    Example:
        with live_log_redirect("logs/job_1.log"):
            convert_pointcloud_to_dem(...)
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the log file in line-buffered mode (buffering=1)
    f = open(log_path, "w", encoding="utf-8", buffering=1)

    # Save original stdout/stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr

    # Redirect both stdout and stderr
    sys.stdout, sys.stderr = f, f

    try:
        yield f
    finally:
        # Flush and restore streams
        f.flush()
        sys.stdout, sys.stderr = old_stdout, old_stderr
        f.close()
