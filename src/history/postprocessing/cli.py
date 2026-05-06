import argparse
import shutil
import sys
from pathlib import Path


_NOTEBOOK_SRC = (
    Path(__file__).parent.parent.parent.parent
    / "notebooks"
    / "postprocessing"
    / "post_process_workflow.ipynb"
)


def _write_slurm_script(output_dir: Path, notebook_dest: Path, python_exec: Path) -> Path:
    py_script = notebook_dest.with_suffix(".py")
    jupytext_exec = python_exec.parent / "jupytext"
    script_path = output_dir / "run.slurm.sh"

    script_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=postprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output={output_dir}/postprocessing.%j.out
#SBATCH --error={output_dir}/postprocessing.%j.out

# Convert notebook to Python script
{jupytext_exec} --to py:percent {notebook_dest}

# Run the script
{python_exec} {py_script}
"""
    )
    return script_path


def main() -> None:
    parser = argparse.ArgumentParser(description="History postprocessing pipeline")
    parser.add_argument("output_dir", help="Output directory for processed results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    notebook_dest = output_dir / _NOTEBOOK_SRC.name
    shutil.copy2(_NOTEBOOK_SRC, notebook_dest)

    python_exec = Path(sys.executable)
    slurm_script = _write_slurm_script(output_dir, notebook_dest, python_exec)

    print("Setup complete. Follow these steps before submitting the job:\n")
    print(f"  1. Open the notebook and adapt the configuration to your data:")
    print(f"       {notebook_dest}\n")
    print(f"  2. Review the SLURM script and adjust resources if needed")
    print(f"     (memory, time limit, partition, etc.):")
    print(f"       {slurm_script}\n")
    print(f"  3. Once everything looks good, submit the job:")
    print(f"       sbatch {slurm_script}")


if __name__ == "__main__":
    main()
