import copy
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.llm_config import call_llm


# ============================================================
# Configuration
# ============================================================

MAX_ITERS = 5
MAX_BUNDLES_FROM_LLM = 20
MAX_MOVES_PER_BUNDLE = 5

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PROMPT_TEMPLATE_PATH = (
    PROJECT_DIR / "final_prompts" / "improvement_prompt.txt"
)
INSTANCE_PATH = PROJECT_DIR / "json_output" / "response1.json"
INITIAL_ROSTER_DIR = PROJECT_DIR / "initial_roster"

OUTPUT_DIR = PROJECT_DIR / "Improved_roster"
BEST_ROSTER_DIR = OUTPUT_DIR / "best_roster"
API_BUNDLES_DIR = OUTPUT_DIR / "api_bundles"
FINAL_VALIDATOR_OUTPUT_FILE = OUTPUT_DIR / "validator_output.txt"

VALIDATOR_DIR = PROJECT_DIR / "validator"
VALIDATOR_WEEKS_DIR = VALIDATOR_DIR / "weeks"
VALIDATOR_JAR = VALIDATOR_DIR / "validator.jar"
VALIDATOR_SCENARIO_FILE = VALIDATOR_DIR / "scenario.json"
VALIDATOR_HISTORY_FILE = VALIDATOR_DIR / "history.json"

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_INDEX = {day: index for index, day in enumerate(DAY_ORDER)}

SHIFT_IMPORT_MAP = {
    "E": "E",
    "L": "L",
    "N": "N",
    "Early": "E",
    "Late": "L",
    "Night": "N",
}

SHIFT_EXPORT_MAP = {
    "E": "Early",
    "L": "Late",
    "N": "Night",
    "Early": "Early",
    "Late": "Late",
    "Night": "Night",
}

Roster = Dict[str, Dict[str, Optional[Tuple[str, str]]]]
SearchHistory = List[Dict[str, Any]]


# ============================================================
# File and path helpers
# ============================================================


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory not found: {path}")


def prepare_output_directory() -> None:
    """Clear previous persistent output and create a clean output folder."""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_week_from_filename(path: Path | str) -> int:
    match = re.search(r"week(\d+)\.json$", str(path).replace("\\", "/"))
    if not match:
        raise ValueError(f"Invalid week filename: {path}")
    return int(match.group(1))


def parse_day_label(day_label: str) -> Tuple[int, str]:
    match = re.fullmatch(
        r"W(\d+)_(Mon|Tue|Wed|Thu|Fri|Sat|Sun)",
        day_label,
    )
    if not match:
        raise ValueError(f"Invalid day label: {day_label}")
    return int(match.group(1)) - 1, match.group(2)


def week_day_to_label(week_zero_based: int, day_short: str) -> str:
    return f"W{week_zero_based + 1}_{day_short}"


def infer_scenario_name(instance: Dict[str, Any]) -> str:
    for key in ("scenario", "name", "instance_name"):
        if key in instance:
            return str(instance[key])
    return "unknown_scenario"


# ============================================================
# Input loading
# ============================================================


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_initial_roster(roster_dir: Path) -> Roster:
    week_files = sorted(
        roster_dir.glob("week*.json"),
        key=parse_week_from_filename,
    )

    if not week_files:
        raise FileNotFoundError(
            f"No initial roster files found in {roster_dir}"
        )

    filename_numbers = [parse_week_from_filename(path) for path in week_files]
    filenames_are_zero_based = min(filename_numbers) == 0

    roster: Roster = {}

    for week_file in week_files:
        with week_file.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {week_file}")

        if "week" in payload:
            week_zero = int(payload["week"])
        else:
            file_week = parse_week_from_filename(week_file)
            week_zero = file_week if filenames_are_zero_based else file_week - 1

        assignments = payload.get("assignments", [])
        if not isinstance(assignments, list):
            raise ValueError(f"'assignments' must be a list in {week_file}")

        for assignment in assignments:
            if not isinstance(assignment, dict):
                raise ValueError(f"Invalid assignment in {week_file}")

            nurse = assignment["nurse"]
            day_short = assignment["day"]
            shift_full = assignment["shiftType"]
            skill = assignment["skill"]

            if day_short not in DAY_INDEX:
                raise ValueError(
                    f"Invalid day '{day_short}' in {week_file}"
                )

            shift = SHIFT_IMPORT_MAP.get(shift_full)
            if shift is None:
                raise ValueError(
                    f"Unknown shift type '{shift_full}' in {week_file}"
                )

            day_label = week_day_to_label(week_zero, day_short)
            roster.setdefault(nurse, {})[day_label] = (shift, skill)

    return roster


def normalize_roster(
    instance: Dict[str, Any],
    partial_roster: Roster,
) -> Roster:
    nurses = instance["sets"]["N"]
    days = instance["sets"]["D"]

    return {
        nurse: {
            day: partial_roster.get(nurse, {}).get(day)
            for day in days
        }
        for nurse in nurses
    }


# ============================================================
# LLM prompt and response handling
# ============================================================


def build_llm_prompt_from_template(
    template_text: str,
    instance: Dict[str, Any],
    roster: Roster,
    validator_output: str,
    search_history: SearchHistory,
) -> str:
    """
    Build the prompt using:
    - the complete instance
    - the current roster
    - the complete raw validator output for that roster
    - all previous bundle evaluation results
    """
    history_text = json.dumps(search_history, indent=2)

    replacements = {
        "{{MAX_BUNDLES}}": str(MAX_BUNDLES_FROM_LLM),
        "{{MAX_MOVES_PER_BUNDLE}}": str(MAX_MOVES_PER_BUNDLE),
        "{{INSTANCE_JSON}}": json.dumps(instance, indent=2),
        "{{CURRENT_ROSTER}}": json.dumps(roster, indent=2),
        "{{VALIDATOR_OUTPUT}}": validator_output,
        "{{SEARCH_HISTORY}}": history_text,
        # Backward compatibility if the old placeholder still exists.
        "{{VALIDATOR_SUMMARY}}": validator_output,
    }

    prompt = template_text
    for placeholder, value in replacements.items():
        prompt = prompt.replace(placeholder, value)

    return prompt


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Extract a direct JSON list or an object containing a 'bundles' list."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    candidates: List[Any] = []

    try:
        candidates.append(json.loads(cleaned))
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end > start:
            try:
                candidates.append(json.loads(cleaned[start : end + 1]))
            except json.JSONDecodeError:
                pass

    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]

        if (
            isinstance(candidate, dict)
            and isinstance(candidate.get("bundles"), list)
        ):
            return [
                item
                for item in candidate["bundles"]
                if isinstance(item, dict)
            ]

    return []


def clean_bundles(bundles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []

    for index, bundle in enumerate(
        bundles[:MAX_BUNDLES_FROM_LLM],
        start=1,
    ):
        moves = bundle.get("moves")
        if not isinstance(moves, list) or not moves:
            continue

        cleaned_bundle = copy.deepcopy(bundle)
        cleaned_bundle["bundle_id"] = bundle.get("bundle_id", index)
        cleaned_bundle["moves"] = moves[:MAX_MOVES_PER_BUNDLE]
        cleaned.append(cleaned_bundle)

    return cleaned


def request_api_bundles(
    instance: Dict[str, Any],
    roster: Roster,
    validator_output: str,
    search_history: SearchHistory,
) -> List[Dict[str, Any]]:
    template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    prompt = build_llm_prompt_from_template(
        template_text=template,
        instance=instance,
        roster=roster,
        validator_output=validator_output,
        search_history=search_history,
    )

    response_text = call_llm(prompt)
    return clean_bundles(extract_json_array(response_text))


def save_api_bundles(
    api_call_number: int,
    bundles: List[Dict[str, Any]],
) -> None:
    """Save one API call's cleaned bundles in its own JSON file."""
    API_BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
    output_file = API_BUNDLES_DIR / f"call_{api_call_number:03d}.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(bundles, file, indent=2)


# ============================================================
# Move validation and application
# ============================================================


def is_qualified(
    instance: Dict[str, Any],
    nurse: str,
    skill: str,
) -> bool:
    return (
        instance["parameters"]["qualifications"]
        .get(nurse, {})
        .get(skill, 0)
        == 1
    )


def apply_move(
    instance: Dict[str, Any],
    roster: Roster,
    move: Dict[str, Any],
) -> Optional[Roster]:
    if not isinstance(move, dict):
        return None

    trial = copy.deepcopy(roster)
    nurses = set(instance["sets"]["N"])
    days = set(instance["sets"]["D"])
    shifts = set(instance["sets"]["S"])
    skills = set(instance["sets"]["K"])

    try:
        move_type = move["type"]

        if move_type == "reassign":
            required = {
                "type",
                "day",
                "from_nurse",
                "to_nurse",
                "shift",
                "skill",
            }
            if set(move) != required:
                return None

            day = move["day"]
            source = move["from_nurse"]
            destination = move["to_nurse"]
            shift = move["shift"]
            skill = move["skill"]

            if (
                day not in days
                or source not in nurses
                or destination not in nurses
                or source == destination
                or shift not in shifts
                or skill not in skills
                or trial[source][day] != (shift, skill)
                or trial[destination][day] is not None
                or not is_qualified(instance, destination, skill)
            ):
                return None

            trial[source][day] = None
            trial[destination][day] = (shift, skill)
            return trial

        if move_type == "swap":
            required = {"type", "day", "nurse_a", "nurse_b"}
            if set(move) != required:
                return None

            day = move["day"]
            nurse_a = move["nurse_a"]
            nurse_b = move["nurse_b"]

            if (
                day not in days
                or nurse_a not in nurses
                or nurse_b not in nurses
                or nurse_a == nurse_b
            ):
                return None

            assignment_a = trial[nurse_a][day]
            assignment_b = trial[nurse_b][day]

            if assignment_a is None and assignment_b is None:
                return None

            if (
                assignment_a is not None
                and not is_qualified(instance, nurse_b, assignment_a[1])
            ):
                return None

            if (
                assignment_b is not None
                and not is_qualified(instance, nurse_a, assignment_b[1])
            ):
                return None

            trial[nurse_a][day], trial[nurse_b][day] = (
                assignment_b,
                assignment_a,
            )
            return trial

        if move_type == "coverage_fill":
            required = {"type", "day", "shift", "skill", "to_nurse"}
            if set(move) != required:
                return None

            day = move["day"]
            nurse = move["to_nurse"]
            shift = move["shift"]
            skill = move["skill"]

            if (
                day not in days
                or nurse not in nurses
                or shift not in shifts
                or skill not in skills
                or trial[nurse][day] is not None
                or not is_qualified(instance, nurse, skill)
            ):
                return None

            trial[nurse][day] = (shift, skill)
            return trial

        if move_type == "weekend_repair":
            required = {"type", "nurse", "week"}
            if set(move) != required:
                return None

            nurse = move["nurse"]
            week_zero = int(move["week"])
            saturday = week_day_to_label(week_zero, "Sat")
            sunday = week_day_to_label(week_zero, "Sun")

            if (
                nurse not in nurses
                or saturday not in days
                or sunday not in days
            ):
                return None

            saturday_work = trial[nurse][saturday] is not None
            sunday_work = trial[nurse][sunday] is not None

            if not (saturday_work ^ sunday_work):
                return None

            if saturday_work:
                trial[nurse][saturday] = None
            else:
                trial[nurse][sunday] = None

            return trial

    except (KeyError, TypeError, ValueError):
        return None

    return None


def apply_bundle(
    instance: Dict[str, Any],
    roster: Roster,
    bundle: Dict[str, Any],
) -> Optional[Roster]:
    if not isinstance(bundle, dict):
        return None

    moves = bundle.get("moves")
    if not isinstance(moves, list) or not moves:
        return None

    trial = copy.deepcopy(roster)

    for move in moves:
        updated = apply_move(instance, trial, move)
        if updated is None:
            return None
        trial = updated

    return trial


# ============================================================
# Roster export and validator
# ============================================================


def export_weekly_inrc2(
    instance: Dict[str, Any],
    roster: Roster,
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    for old_file in output_directory.glob("week*.json"):
        old_file.unlink()

    days = instance["sets"]["D"]
    nurses = instance["sets"]["N"]
    scenario = infer_scenario_name(instance)
    weeks = sorted({parse_day_label(day)[0] for day in days})

    grouped: Dict[int, List[Dict[str, Any]]] = {
        week: [] for week in weeks
    }

    for nurse in nurses:
        for day in days:
            assignment = roster[nurse][day]
            if assignment is None:
                continue

            week_zero, day_short = parse_day_label(day)
            shift, skill = assignment

            grouped[week_zero].append(
                {
                    "nurse": nurse,
                    "day": day_short,
                    "shiftType": SHIFT_EXPORT_MAP[shift],
                    "skill": skill,
                }
            )

    for week_zero in weeks:
        grouped[week_zero].sort(
            key=lambda assignment: (
                assignment["nurse"],
                DAY_INDEX[assignment["day"]],
                assignment["shiftType"],
                assignment["skill"],
            )
        )

        payload = {
            "scenario": scenario,
            "week": week_zero,
            "assignments": grouped[week_zero],
        }

        output_path = output_directory / f"week{week_zero}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)


def get_validator_week_files() -> List[Path]:
    week_files = sorted(
        VALIDATOR_WEEKS_DIR.glob("week*.json"),
        key=parse_week_from_filename,
    )

    if not week_files:
        raise FileNotFoundError(
            f"No validator week files found in {VALIDATOR_WEEKS_DIR}"
        )

    return week_files


def run_validator(solution_directory: Path) -> str:
    solution_files = sorted(
        solution_directory.glob("week*.json"),
        key=parse_week_from_filename,
    )
    validator_week_files = get_validator_week_files()

    if not solution_files:
        raise FileNotFoundError(
            f"No solution week files found in {solution_directory}"
        )

    if len(solution_files) != len(validator_week_files):
        raise ValueError(
            "Week-file count mismatch: "
            f"solutions={len(solution_files)}, "
            f"validator weeks={len(validator_week_files)}"
        )

    command = [
        "java",
        "-jar",
        str(VALIDATOR_JAR),
        "--sce",
        str(VALIDATOR_SCENARIO_FILE),
        "--his",
        str(VALIDATOR_HISTORY_FILE),
        "--weeks",
        *[str(path) for path in validator_week_files],
        "--sols",
        *[str(path) for path in solution_files],
        "--verbose",
    ]

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        details = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or "unknown validator error"
        )
        raise RuntimeError(f"Validator failed: {details}")

    output = completed.stdout.strip() or completed.stderr.strip()
    if not output:
        raise RuntimeError("Validator returned no output")

    return output


def parse_validator_output(text: str) -> Dict[str, Any]:
    required_labels = {
        "minimal_coverage": "Minimal coverage constraints",
        "required_skill": "Required skill constraints",
        "illegal_succession": "Illegal shift type succession constraints",
        "single_assignment": "Single assignment per day",
        "total_cost": "Total cost",
    }

    optional_labels = {
        "total_assignment_cost": "Total assignment constraints",
        "consecutive_cost": "Consecutive constraints",
        "non_working_days_cost": "Non working days constraints",
        "preferences_cost": "Preferences",
        "max_working_weekend_cost": "Max working weekend",
        "complete_weekends_cost": "Complete weekends",
        "optimal_coverage_cost": "Optimal coverage constraints",
    }

    result: Dict[str, Any] = {}

    for key, label in required_labels.items():
        match = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
        if not match:
            raise ValueError(
                f"Validator output is missing required field: {label}"
            )
        result[key] = int(match.group(1))

    for key, label in optional_labels.items():
        match = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
        result[key] = int(match.group(1)) if match else 0

    result["hard_ok"] = all(
        result[key] == 0
        for key in (
            "minimal_coverage",
            "required_skill",
            "illegal_succession",
            "single_assignment",
        )
    )

    return result


def evaluate_roster(
    instance: Dict[str, Any],
    roster: Roster,
    temporary_root: Path,
    candidate_name: str,
) -> Tuple[str, Dict[str, Any]]:
    candidate_directory = temporary_root / candidate_name
    export_weekly_inrc2(instance, roster, candidate_directory)
    raw_output = run_validator(candidate_directory)
    parsed_output = parse_validator_output(raw_output)
    return raw_output, parsed_output


# ============================================================
# Search
# ============================================================


def result_sort_key(result: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        result["total_cost"],
        result["optimal_coverage_cost"],
        result["total_assignment_cost"],
        result["consecutive_cost"],
    )


def choose_best_bundle(
    instance: Dict[str, Any],
    current_roster: Roster,
    current_result: Dict[str, Any],
    bundles: List[Dict[str, Any]],
    temporary_root: Path,
    iteration: int,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Roster],
    Optional[str],
    Optional[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    best_bundle: Optional[Dict[str, Any]] = None
    best_roster: Optional[Roster] = None
    best_raw: Optional[str] = None
    best_result: Optional[Dict[str, Any]] = None
    best_record_index: Optional[int] = None
    bundle_results: List[Dict[str, Any]] = []

    for bundle_index, bundle in enumerate(bundles, start=1):
        record: Dict[str, Any] = {
            "iteration": iteration,
            "bundle_index": bundle_index,
            "bundle_id": bundle.get("bundle_id", bundle_index),
            "reason": bundle.get("reason", ""),
            "moves": copy.deepcopy(bundle.get("moves", [])),
            "starting_total_cost": current_result["total_cost"],
        }

        trial_roster = apply_bundle(instance, current_roster, bundle)
        if trial_roster is None:
            record["status"] = "precheck_failed"
            record["message"] = "invalid move structure or precheck failed"
            bundle_results.append(record)
            print(
                f"Skipping bundle {record['bundle_id']}: "
                "invalid move structure or precheck failed"
            )
            continue

        try:
            raw_output, parsed_result = evaluate_roster(
                instance,
                trial_roster,
                temporary_root,
                f"iteration_{iteration}_bundle_{bundle_index}",
            )
        except Exception as error:
            record["status"] = "validator_error"
            record["message"] = str(error)
            bundle_results.append(record)
            print(f"Skipping bundle {record['bundle_id']}: {error}")
            continue

        record["validator_result"] = copy.deepcopy(parsed_result)
        record["cost_change"] = (
            parsed_result["total_cost"] - current_result["total_cost"]
        )

        if not parsed_result["hard_ok"]:
            record["status"] = "hard_infeasible"
            bundle_results.append(record)
            continue

        if parsed_result["total_cost"] >= current_result["total_cost"]:
            record["status"] = "feasible_not_improving"
            bundle_results.append(record)
            continue

        record["status"] = "improving_candidate"
        bundle_results.append(record)

        if (
            best_result is None
            or result_sort_key(parsed_result) < result_sort_key(best_result)
        ):
            best_bundle = bundle
            best_roster = trial_roster
            best_raw = raw_output
            best_result = parsed_result
            best_record_index = len(bundle_results) - 1

    if best_record_index is not None and best_result is not None:
        bundle_results[best_record_index]["status"] = "accepted"
        bundle_results[best_record_index]["accepted_total_cost"] = best_result[
            "total_cost"
        ]

    return (
        best_bundle,
        best_roster,
        best_raw,
        best_result,
        bundle_results,
    )


# ============================================================
# Final saving
# ============================================================


def save_final_result(
    instance: Dict[str, Any],
    roster: Roster,
    validator_output: str,
) -> None:
    export_weekly_inrc2(instance, roster, BEST_ROSTER_DIR)
    FINAL_VALIDATOR_OUTPUT_FILE.write_text(
        validator_output,
        encoding="utf-8",
    )


# ============================================================
# Main pipeline
# ============================================================


def run_pipeline() -> None:
    require_file(PROMPT_TEMPLATE_PATH)
    require_file(INSTANCE_PATH)
    require_directory(INITIAL_ROSTER_DIR)
    require_file(VALIDATOR_JAR)
    require_file(VALIDATOR_SCENARIO_FILE)
    require_file(VALIDATOR_HISTORY_FILE)
    require_directory(VALIDATOR_WEEKS_DIR)

    instance = load_json(INSTANCE_PATH)
    initial_partial_roster = load_initial_roster(INITIAL_ROSTER_DIR)
    current_roster = normalize_roster(instance, initial_partial_roster)

    with tempfile.TemporaryDirectory(prefix="roster_improvement_") as temp_dir:
        temporary_root = Path(temp_dir)

        # This is the complete validator output for the initial roster.
        current_raw, current_result = evaluate_roster(
            instance,
            current_roster,
            temporary_root,
            "initial",
        )

        if not current_result["hard_ok"]:
            raise RuntimeError("Initial roster is not hard-feasible")

        prepare_output_directory()

        best_roster = copy.deepcopy(current_roster)
        best_raw = current_raw
        best_result = current_result

        # Kept in memory and inserted into every later prompt.
        search_history: SearchHistory = []

        for iteration in range(1, MAX_ITERS + 1):
            try:
                # The prompt receives:
                # 1. the current roster
                # 2. the complete validator output for that roster
                # 3. all prior bundle results
                bundles = request_api_bundles(
                    instance=instance,
                    roster=current_roster,
                    validator_output=current_raw,
                    search_history=search_history,
                )
            except Exception as error:
                print(f"Iteration {iteration}: API call failed: {error}")
                bundles = []

            save_api_bundles(
                api_call_number=iteration,
                bundles=bundles,
            )

            if not bundles:
                search_history.append(
                    {
                        "iteration": iteration,
                        "status": "api_returned_no_valid_bundles",
                    }
                )
                print(f"Iteration {iteration}: API returned no valid bundles")
                break

            (
                accepted_bundle,
                next_roster,
                next_raw,
                next_result,
                iteration_results,
            ) = choose_best_bundle(
                instance=instance,
                current_roster=current_roster,
                current_result=current_result,
                bundles=bundles,
                temporary_root=temporary_root,
                iteration=iteration,
            )

            # These results are included in the next iteration's prompt.
            search_history.extend(iteration_results)

            if (
                accepted_bundle is None
                or next_roster is None
                or next_raw is None
                or next_result is None
            ):
                print(f"Iteration {iteration}: no improving valid bundle found")
                continue

            # CRITICAL: the accepted candidate becomes the current roster.
            # The next API call therefore sees the accepted roster and its
            # complete validator output, not the original roster.
            current_roster = next_roster
            current_raw = next_raw
            current_result = next_result

            if current_result["total_cost"] < best_result["total_cost"]:
                best_roster = copy.deepcopy(current_roster)
                best_raw = current_raw
                best_result = copy.deepcopy(current_result)

            print(
                f"Iteration {iteration}: accepted bundle "
                f"{accepted_bundle.get('bundle_id', '?')}, "
                f"cost {current_result['total_cost']}"
            )

        save_final_result(
            instance=instance,
            roster=best_roster,
            validator_output=best_raw,
        )

        print(f"Final best roster cost: {best_result['total_cost']}")
        print(f"Saved API calls to: {API_BUNDLES_DIR}")
        print(f"Saved validator output to: {FINAL_VALIDATOR_OUTPUT_FILE}")
        print(f"Saved best roster to: {BEST_ROSTER_DIR}")


def main() -> None:
    try:
        run_pipeline()
    except Exception as error:
        print(f"Roster improvement failed: {error}")


if __name__ == "__main__":
    main()