import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


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
