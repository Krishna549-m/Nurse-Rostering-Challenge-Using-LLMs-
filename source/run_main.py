import subprocess
import sys
from pathlib import Path

PYTHON_EXE = sys.executable

RUN_PROMPT_FILE = "source/run_prompt.py"
IMPROVEMENT_FILE = "source/improvement_phase1.py"

VALIDATOR_JAR = "validator/validator.jar"
SCENARIO_FILE = "validator/scenario.json"
HISTORY_FILE = "validator/history.json"

WEEK_FILES = [
    "validator/weeks/week1.json",
    "validator/weeks/week2.json",
    "validator/weeks/week3.json",
    "validator/weeks/week4.json",
]

SOLUTION_FILES = [
    "initial_roster/week0.json",
    "initial_roster/week1.json",
    "initial_roster/week2.json",
    "initial_roster/week3.json",
]

WORKING_DIR = "."  


def run_command(cmd, step_name, cwd=None):
    print(f"\n{'=' * 70}")
    print(f"Running: {step_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd, cwd=cwd, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}")

    print(f"\n{step_name} completed successfully.\n")


def check_files_exist(file_list):
    missing = [str(f) for f in file_list if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files were not found:\n"
            + "\n".join(missing)
            + "\n\nAll checked files were:\n"
            + "\n".join(str(f) for f in file_list)
        )

def main():
    base = Path(__file__).resolve().parents[1]
    print(f"Working directory: {base}")

    run_prompt_path = base / RUN_PROMPT_FILE
    improvement_path = base / IMPROVEMENT_FILE
    validator_path = base / VALIDATOR_JAR
    scenario_path = base / SCENARIO_FILE
    history_path = base / HISTORY_FILE
    week_paths = [base / f for f in WEEK_FILES]
    solution_paths = [base / f for f in SOLUTION_FILES]

    check_files_exist(
        [run_prompt_path, improvement_path, validator_path, scenario_path, history_path]
        + week_paths
    )

    # Run run_prompt.py
    run_command(
        [PYTHON_EXE, str(run_prompt_path)],
        "Step 1 - run_prompt.py",
        cwd=str(base),
    )

    # Run solver.py
    run_command(
    [PYTHON_EXE, str(base / "source/solver.py")],
    "Step 2 - solver.py",
    cwd=str(base),
    )

    # Step 2: Run validator
    validator_cmd = [
        "java",
        "-jar",
        str(validator_path),
        "--sce",
        str(scenario_path),
        "--his",
        str(history_path),
        "--weeks",
        *[str(p) for p in week_paths],
        "--sols",
        *[str(p) for p in solution_paths],
        "--verbose",
    ]

    with open(base / "initial_roster/validator_output.txt", "w", encoding="utf-8") as f:
        result = subprocess.run(
            validator_cmd,
            cwd=str(base),
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT
        )

    if result.returncode != 0:
        raise RuntimeError("Step 3 - INRC-II Validator failed")

    # Run improvement.py
    run_command(
        [PYTHON_EXE, str(improvement_path)],
        "Step 4 - improvement.py",
        cwd=str(base),
    )

    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()