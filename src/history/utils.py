import logging
import shutil
from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Yield a temporary sibling of `path` to write to; it replaces `path` only if the block succeeds.

    Works for files and directories.
    """
    path = Path(path)
    # keep the suffix so rasterio / savefig still infer the format from the name
    tmp = path.with_name(f".tmp-{path.name}")

    # leftover from a previous run killed before cleanup
    _remove(tmp)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        yield tmp
    except BaseException:
        _remove(tmp)
        raise
    _remove(path)
    tmp.replace(path)


def _remove(path: Path) -> None:
    """Delete a file or directory tree if it exists."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


@contextmanager
def log_to_file(log_path: str | Path, logger: logging.Logger):
    """
    Context manager adding a temporary file handler to a logger.

    Parameters
    ----------
    log_path : str | Path
        - If a directory → a timestamped log file will be created inside it
        - If a file path → this exact file will be used
    """
    log_path = Path(log_path)

    # If a directory is provided, generate a timestamped filename
    if log_path.is_dir() or not log_path.suffix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_path / f"log_{timestamp}.log"

    # Ensure parent directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    # Register handler
    logger.addHandler(file_handler)

    try:
        yield log_path  # optionally return final path
    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


def is_output_up_to_date(inputs: str | Path | Sequence[str | Path], outputs: str | Path | Sequence[str | Path]) -> bool:
    """Return True if all outputs exist and are newer than all inputs."""
    inputs = [inputs] if isinstance(inputs, (str, Path)) else inputs
    outputs = [outputs] if isinstance(outputs, (str, Path)) else outputs

    # standardize path
    inputs = [Path(f) for f in inputs]
    outputs = [Path(f) for f in outputs]

    if not inputs:
        raise ValueError("inputs cannot be empty")
    if not outputs:
        raise ValueError("outputs cannot be empty")

    if any(not p.exists() for p in outputs):
        return False

    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    oldest_output = min(p.stat().st_mtime for p in outputs)
    newest_input = max(p.stat().st_mtime for p in inputs)

    return newest_input <= oldest_output


def skip_overwrite(
    inputs: str | Path | Sequence[str | Path] | None, outputs: str | Path | Sequence[str | Path], overwrite: bool
) -> bool:
    """Return True (and log it) if the outputs are up to date and `overwrite` is not set.

    If `inputs` is None or empty, only the existence of the outputs is checked.
    """
    if overwrite:
        return False

    files = [outputs] if isinstance(outputs, (str, Path)) else list(outputs)
    if inputs is None or not isinstance(inputs, (str, Path)) and len(inputs) == 0:
        up_to_date = all(Path(f).exists() for f in files)
    else:
        up_to_date = is_output_up_to_date(inputs, files)

    if up_to_date:
        label = files[0] if len(files) == 1 else f"{files[0]} (+{len(files) - 1} more)"
        logger.info("%s - already up to date, skipped (use overwrite to redo)", label)
    return up_to_date


@contextmanager
def write_file(output_path: str | Path) -> Generator[Path, None, None]:
    """Create the parent directory of `output_path`, yield it, and log once the file has been written."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yield output_path
    logger.info(f"Written {output_path}")
