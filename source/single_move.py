import os
import re
import csv
import json
import copy
import glob
import shutil
import random
import subprocess
from typing import Dict, List, Tuple, Optional, Any


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
MAX_ITERS = 2

ENABLE_REASSIGN_MOVES = True
ENABLE_SWAP_MOVES = True
ENABLE_REMOVE_ASSIGNMENT_MOVES = True
ENABLE_ADD_ASSIGNMENT_MOVES = True

INSTANCE_PATH = "../json_output/response1.json"

INITIAL_ROSTER_DIR = "../initial_roster"
INITIAL_VALIDATOR_OUTPUT_PATH = os.path.join(INITIAL_ROSTER_DIR, "validator_output.txt")

OUTPUT_DIR = "improved_roster_deterministic"
TMP_DIR = "tmp_candidates_deterministic"
VALIDATOR_CALLS_DIR = os.path.join(OUTPUT_DIR, "validator_calls")

VALIDATOR_DIR = "../validator"
VALIDATOR_WEEKS_DIR = os.path.join(VALIDATOR_DIR, "weeks")

VALIDATOR_JAR = os.path.join(VALIDATOR_DIR, "validator.jar")
VALIDATOR_SCENARIO_FILE = os.path.join(VALIDATOR_DIR, "scenario.json")
VALIDATOR_HISTORY_FILE = os.path.join(VALIDATOR_DIR, "history.json")

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_INDEX = {d: i for i, d in enumerate(DAY_ORDER)}

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

random.seed(RANDOM_SEED)


# ============================================================
# Basic helpers
# ============================================================

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(VALIDATOR_CALLS_DIR, exist_ok=True)


def safe_delete_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def deep_copy_roster(roster):
    return copy.deepcopy(roster)


def load_text_if_exists(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_day_label(day_label: str) -> Tuple[int, str]:
    m = re.match(r"W(\d+)_(Mon|Tue|Wed|Thu|Fri|Sat|Sun)$", day_label)
    if not m:
        raise ValueError(f"Invalid day label: {day_label}")
    return int(m.group(1)) - 1, m.group(2)


def week_day_to_label(week_zero_based: int, day_short: str) -> str:
    return f"W{week_zero_based + 1}_{day_short}"


def parse_week_from_filename(path: str) -> int:
    m = re.search(r"week(\d+)\.json$", path.replace("\\", "/"))
    if not m:
        raise ValueError(f"Invalid week filename: {path}")
    return int(m.group(1))


def infer_scenario_name(instance: Dict[str, Any]) -> str:
    for key in ("scenario", "name", "instance_name"):
        if key in instance:
            return str(instance[key])
    return "unknown_scenario"


# ============================================================
# Load instance and roster
# ============================================================

def load_instance(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_initial_roster(
    roster_dir: str,
) -> Dict[str, Dict[str, Optional[Tuple[str, str]]]]:
    week_files = sorted(
        glob.glob(os.path.join(roster_dir, "week*.json")),
        key=parse_week_from_filename,
    )

    if not week_files:
        raise FileNotFoundError("No valid initial_roster/week*.json files found")

    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]] = {}

    for week_file in week_files:
        with open(week_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        week_num = payload.get("week")
        if week_num is None:
            week_num = parse_week_from_filename(week_file) - 1

        assignments = payload.get("assignments", [])

        for a in assignments:
            nurse = a["nurse"]
            day_short = a["day"]
            shift_full = a["shiftType"]
            skill = a["skill"]

            if day_short not in DAY_INDEX:
                raise ValueError(f"Invalid day in {week_file}: {day_short}")

            shift = SHIFT_IMPORT_MAP.get(shift_full)
            if shift is None:
                raise ValueError(f"Unknown shiftType in {week_file}: {shift_full}")

            day_label = f"W{week_num + 1}_{day_short}"
            roster.setdefault(nurse, {})[day_label] = (shift, skill)

    return roster


def normalize_roster(
    instance: Dict[str, Any],
    partial_roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> Dict[str, Dict[str, Optional[Tuple[str, str]]]]:
    nurses = instance["sets"]["N"]
    days = instance["sets"]["D"]

    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]] = {}

    for nurse in nurses:
        roster[nurse] = {}
        for day in days:
            roster[nurse][day] = partial_roster.get(nurse, {}).get(day)

    return roster


# ============================================================
# Export roster to INRC-II solution files
# ============================================================

def export_weekly_inrc2(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    days = instance["sets"]["D"]
    nurses = instance["sets"]["N"]
    scenario = infer_scenario_name(instance)

    weeks = sorted({parse_day_label(d)[0] for d in days})
    grouped: Dict[int, List[Dict[str, Any]]] = {w: [] for w in weeks}

    for nurse in nurses:
        for day in days:
            assignment = roster[nurse][day]

            if assignment is None:
                continue

            week_zero, day_short = parse_day_label(day)
            shift, skill = assignment

            grouped[week_zero].append({
                "nurse": nurse,
                "day": day_short,
                "shiftType": SHIFT_EXPORT_MAP[shift],
                "skill": skill,
            })

    for week_zero in weeks:
        grouped[week_zero].sort(
            key=lambda a: (
                a["nurse"],
                DAY_INDEX[a["day"]],
                a["shiftType"],
                a["skill"],
            )
        )

        payload = {
            "scenario": scenario,
            "week": week_zero,
            "assignments": grouped[week_zero],
        }

        out_path = os.path.join(out_dir, f"week{week_zero + 1}.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


# ============================================================
# INRC-II validator interaction
# ============================================================

def parse_week_index_from_validator_week_file(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"week(\d+)\.json$", name)
    if not m:
        raise ValueError(f"Could not parse week index from validator week file: {path}")
    return int(m.group(1))


def get_validator_week_files() -> List[str]:
    week_files = glob.glob(os.path.join(VALIDATOR_WEEKS_DIR, "*.json"))

    if not week_files:
        raise FileNotFoundError(f"No validator week files found in {VALIDATOR_WEEKS_DIR}")

    return sorted(week_files, key=parse_week_index_from_validator_week_file)


def validator_command_for_solution_dir(solution_dir: str) -> List[str]:
    sol_files = sorted(
        glob.glob(os.path.join(solution_dir, "week*.json")),
        key=parse_week_from_filename,
    )

    if not sol_files:
        raise FileNotFoundError(f"No week*.json files in {solution_dir}")

    validator_week_files = get_validator_week_files()

    if len(sol_files) != len(validator_week_files):
        raise ValueError(
            f"Mismatch between solution week files ({len(sol_files)}) "
            f"and validator week files ({len(validator_week_files)})"
        )

    cmd = [
        "java",
        "-jar",
        VALIDATOR_JAR,
        "--sce",
        VALIDATOR_SCENARIO_FILE,
        "--his",
        VALIDATOR_HISTORY_FILE,
        "--weeks",
        *validator_week_files,
        "--sols",
        *sol_files,
        "--verbose",
    ]

    return cmd


def run_validator(solution_dir: str) -> str:
    cmd = validator_command_for_solution_dir(solution_dir)

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0 and not completed.stdout:
        error_text = completed.stderr.strip() or "unknown validator error"
        raise RuntimeError(f"Validator failed: {error_text}")

    return completed.stdout if completed.stdout else completed.stderr


def parse_validator_output(text: str) -> Dict[str, Any]:
    def extract_int(label: str, default: int = 0) -> int:
        m = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
        return int(m.group(1)) if m else default

    result = {
        "minimal_coverage": extract_int("Minimal coverage constraints"),
        "required_skill": extract_int("Required skill constraints"),
        "illegal_succession": extract_int("Illegal shift type succession constraints"),
        "single_assignment": extract_int("Single assignment per day"),
        "total_assignment_cost": extract_int("Total assignment constraints"),
        "consecutive_cost": extract_int("Consecutive constraints"),
        "non_working_days_cost": extract_int("Non working days constraints"),
        "preferences_cost": extract_int("Preferences"),
        "max_working_weekend_cost": extract_int("Max working weekend"),
        "complete_weekends_cost": extract_int("Complete weekends"),
        "optimal_coverage_cost": extract_int("Optimal coverage constraints"),
        "total_cost": extract_int("Total cost", default=10**12),
    }

    result["hard_ok"] = (
        result["minimal_coverage"] == 0
        and result["required_skill"] == 0
        and result["illegal_succession"] == 0
        and result["single_assignment"] == 0
    )

    return result


def evaluate_roster_with_validator(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    candidate_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    safe_delete_dir(candidate_dir)
    os.makedirs(candidate_dir, exist_ok=True)

    export_weekly_inrc2(instance, roster, candidate_dir)

    raw = run_validator(candidate_dir)
    parsed = parse_validator_output(raw)

    return raw, parsed


# ============================================================
# Move application
# ============================================================

def apply_move(
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    move: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]]:
    if not isinstance(move, dict):
        return None

    trial = deep_copy_roster(roster)
    move_type = move.get("type")

    try:
        if move_type == "reassign":
            required_keys = {"type", "day", "from_nurse", "to_nurse", "shift", "skill"}
            if not required_keys.issubset(move.keys()):
                return None

            day = move["day"]
            from_nurse = move["from_nurse"]
            to_nurse = move["to_nurse"]
            shift = move["shift"]
            skill = move["skill"]

            if trial[from_nurse][day] != (shift, skill):
                return None

            if trial[to_nurse][day] is not None:
                return None

            trial[from_nurse][day] = None
            trial[to_nurse][day] = (shift, skill)

            return trial

        if move_type == "swap":
            required_keys = {"type", "day", "nurse_a", "nurse_b"}
            if not required_keys.issubset(move.keys()):
                return None

            day = move["day"]
            nurse_a = move["nurse_a"]
            nurse_b = move["nurse_b"]

            if trial[nurse_a][day] is None and trial[nurse_b][day] is None:
                return None

            trial[nurse_a][day], trial[nurse_b][day] = (
                trial[nurse_b][day],
                trial[nurse_a][day],
            )

            return trial

        if move_type == "remove_assignment":
            required_keys = {"type", "day", "nurse"}
            if not required_keys.issubset(move.keys()):
                return None

            day = move["day"]
            nurse = move["nurse"]

            if trial[nurse][day] is None:
                return None

            trial[nurse][day] = None

            return trial

        if move_type == "add_assignment":
            required_keys = {"type", "day", "nurse", "shift", "skill"}
            if not required_keys.issubset(move.keys()):
                return None

            day = move["day"]
            nurse = move["nurse"]
            shift = move["shift"]
            skill = move["skill"]

            if trial[nurse][day] is not None:
                return None

            trial[nurse][day] = (shift, skill)

            return trial

    except Exception:
        return None

    return None


def apply_bundle(
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    bundle: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]]:
    if not isinstance(bundle, dict):
        return None

    moves = bundle.get("moves")

    if not isinstance(moves, list) or len(moves) == 0:
        return None

    trial = deep_copy_roster(roster)

    for move in moves:
        next_trial = apply_move(trial, move)
        if next_trial is None:
            return None
        trial = next_trial

    return trial


# ============================================================
# Deterministic candidate generation
# ============================================================

def generate_reassign_bundles(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> List[Dict[str, Any]]:
    bundles = []

    nurses = instance["sets"]["N"]
    days = instance["sets"]["D"]
    qualifications = instance["parameters"]["qualifications"]

    for from_nurse in nurses:
        for day in days:
            assignment = roster[from_nurse][day]

            if assignment is None:
                continue

            shift, skill = assignment

            for to_nurse in nurses:
                if to_nurse == from_nurse:
                    continue

                if roster[to_nurse][day] is not None:
                    continue

                if qualifications[to_nurse].get(skill, 0) != 1:
                    continue

                bundles.append({
                    "bundle_id": 0,
                    "reason": "deterministic single reassign candidate",
                    "moves": [
                        {
                            "type": "reassign",
                            "day": day,
                            "from_nurse": from_nurse,
                            "to_nurse": to_nurse,
                            "shift": shift,
                            "skill": skill,
                        }
                    ],
                })

    return bundles


def generate_swap_bundles(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> List[Dict[str, Any]]:
    bundles = []

    nurses = instance["sets"]["N"]
    days = instance["sets"]["D"]

    for day in days:
        for i in range(len(nurses)):
            for j in range(i + 1, len(nurses)):
                nurse_a = nurses[i]
                nurse_b = nurses[j]

                if roster[nurse_a][day] is None and roster[nurse_b][day] is None:
                    continue

                bundles.append({
                    "bundle_id": 0,
                    "reason": "deterministic single swap candidate",
                    "moves": [
                        {
                            "type": "swap",
                            "day": day,
                            "nurse_a": nurse_a,
                            "nurse_b": nurse_b,
                        }
                    ],
                })

    return bundles


def generate_remove_assignment_bundles(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> List[Dict[str, Any]]:
    bundles = []

    nurses = instance["sets"]["N"]
    days = instance["sets"]["D"]

    for nurse in nurses:
        for day in days:
            if roster[nurse][day] is None:
                continue

            bundles.append({
                "bundle_id": 0,
                "reason": "deterministic single remove-assignment candidate",
                "moves": [
                    {
                        "type": "remove_assignment",
                        "day": day,
                        "nurse": nurse,
                    }
                ],
            })

    return bundles


def generate_add_assignment_bundles(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> List[Dict[str, Any]]:
    bundles = []

    nurses = instance["sets"]["N"]
    days = instance["sets"]["D"]
    shifts = instance["sets"]["S"]
    skills = instance["sets"]["K"]
    qualifications = instance["parameters"]["qualifications"]

    for nurse in nurses:
        for day in days:
            if roster[nurse][day] is not None:
                continue

            for shift in shifts:
                for skill in skills:
                    if qualifications[nurse].get(skill, 0) != 1:
                        continue

                    bundles.append({
                        "bundle_id": 0,
                        "reason": "deterministic single add-assignment candidate",
                        "moves": [
                            {
                                "type": "add_assignment",
                                "day": day,
                                "nurse": nurse,
                                "shift": shift,
                                "skill": skill,
                            }
                        ],
                    })

    return bundles


def generate_deterministic_single_move_bundles(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> List[Dict[str, Any]]:
    bundles = []

    if ENABLE_REASSIGN_MOVES:
        bundles.extend(generate_reassign_bundles(instance, roster))

    if ENABLE_SWAP_MOVES:
        bundles.extend(generate_swap_bundles(instance, roster))

    if ENABLE_REMOVE_ASSIGNMENT_MOVES:
        bundles.extend(generate_remove_assignment_bundles(instance, roster))

    if ENABLE_ADD_ASSIGNMENT_MOVES:
        bundles.extend(generate_add_assignment_bundles(instance, roster))

    for i, bundle in enumerate(bundles, start=1):
        bundle["bundle_id"] = i

    return bundles


# ============================================================
# Logging
# ============================================================

def diff_rosters(
    before: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    after: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> List[str]:
    changes = []

    for nurse in sorted(before.keys()):
        for day in sorted(
            before[nurse].keys(),
            key=lambda d: (parse_day_label(d)[0], DAY_INDEX[parse_day_label(d)[1]]),
        ):
            old = before[nurse][day]
            new = after[nurse][day]

            if old != new:
                old_str = "OFF" if old is None else f"{old[0]}/{old[1]}"
                new_str = "OFF" if new is None else f"{new[0]}/{new[1]}"
                changes.append(f"{day}: {nurse} {old_str} -> {new_str}")

    return changes


def save_validator_call(
    iteration: int,
    bundle_index: int,
    raw_text: str,
    parsed: Dict[str, Any],
    bundle: Dict[str, Any],
) -> None:
    base = f"call_{iteration:04d}_bundle_{bundle_index:04d}"

    txt_path = os.path.join(VALIDATOR_CALLS_DIR, f"{base}.txt")
    json_path = os.path.join(VALIDATOR_CALLS_DIR, f"{base}.json")
    bundle_path = os.path.join(VALIDATOR_CALLS_DIR, f"{base}_bundle.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)


def save_search_log(log_data: Dict[str, Any]) -> None:
    path = os.path.join(OUTPUT_DIR, "search_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


def save_validator_calls_index(rows: List[Dict[str, Any]]) -> None:
    path = os.path.join(OUTPUT_DIR, "validator_calls_index.csv")

    fieldnames = [
        "call_id",
        "iteration",
        "bundle_index",
        "bundle_id",
        "move_type",
        "hard_ok",
        "old_total_cost",
        "new_total_cost",
        "delta",
        "minimal_coverage",
        "required_skill",
        "illegal_succession",
        "single_assignment",
        "total_assignment_cost",
        "consecutive_cost",
        "non_working_days_cost",
        "preferences_cost",
        "max_working_weekend_cost",
        "complete_weekends_cost",
        "optimal_coverage_cost",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def save_best_outputs(
    instance: Dict[str, Any],
    initial_raw: str,
    best_raw: str,
    best_roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> None:
    export_weekly_inrc2(instance, best_roster, OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "initial_validator_summary.txt"), "w", encoding="utf-8") as f:
        f.write(initial_raw)

    with open(os.path.join(OUTPUT_DIR, "best_validator_summary.txt"), "w", encoding="utf-8") as f:
        f.write(best_raw)


# ============================================================
# Candidate evaluation
# ============================================================

def bundle_sort_key(result: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    return (
        result["total_cost"],
        result["non_working_days_cost"],
        result["optimal_coverage_cost"],
        result["consecutive_cost"],
        result["total_assignment_cost"],
    )


def choose_best_bundle(
    instance: Dict[str, Any],
    current_roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    current_result: Dict[str, Any],
    bundles: List[Dict[str, Any]],
    iteration: int,
    validator_index_rows: List[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]],
    Optional[str],
    Optional[Dict[str, Any]],
]:
    best_bundle = None
    best_roster = None
    best_raw = None
    best_result = None

    old_cost = current_result["total_cost"]

    for bundle_index, bundle in enumerate(bundles, start=1):
        trial_roster = apply_bundle(current_roster, bundle)

        if trial_roster is None:
            continue

        candidate_dir = os.path.join(
            TMP_DIR,
            f"iter_{iteration:04d}_bundle_{bundle_index:04d}",
        )

        try:
            raw, parsed = evaluate_roster_with_validator(instance, trial_roster, candidate_dir)
        except Exception as e:
            print(f"Validator error on bundle {bundle_index}: {e}")
            continue

        delta = parsed["total_cost"] - old_cost
        move_type = bundle["moves"][0].get("type", "unknown") if bundle.get("moves") else "unknown"

        save_validator_call(iteration, bundle_index, raw, parsed, bundle)

        validator_index_rows.append({
            "call_id": f"{iteration:04d}_{bundle_index:04d}",
            "iteration": iteration,
            "bundle_index": bundle_index,
            "bundle_id": bundle.get("bundle_id", bundle_index),
            "move_type": move_type,
            "hard_ok": parsed["hard_ok"],
            "old_total_cost": old_cost,
            "new_total_cost": parsed["total_cost"],
            "delta": delta,
            "minimal_coverage": parsed["minimal_coverage"],
            "required_skill": parsed["required_skill"],
            "illegal_succession": parsed["illegal_succession"],
            "single_assignment": parsed["single_assignment"],
            "total_assignment_cost": parsed["total_assignment_cost"],
            "consecutive_cost": parsed["consecutive_cost"],
            "non_working_days_cost": parsed["non_working_days_cost"],
            "preferences_cost": parsed["preferences_cost"],
            "max_working_weekend_cost": parsed["max_working_weekend_cost"],
            "complete_weekends_cost": parsed["complete_weekends_cost"],
            "optimal_coverage_cost": parsed["optimal_coverage_cost"],
        })

        if not parsed["hard_ok"]:
            continue

        if parsed["total_cost"] < current_result["total_cost"]:
            if best_result is None or bundle_sort_key(parsed) < bundle_sort_key(best_result):
                best_bundle = bundle
                best_roster = trial_roster
                best_raw = raw
                best_result = parsed

    return best_bundle, best_roster, best_raw, best_result


# ============================================================
# Main deterministic local-search pipeline
# ============================================================

def run_deterministic_improvement() -> None:
    ensure_dirs()

    if not os.path.exists(INSTANCE_PATH):
        print(f"Missing instance file: {INSTANCE_PATH}")
        return

    try:
        instance = load_instance(INSTANCE_PATH)
        initial_partial_roster = load_initial_roster(INITIAL_ROSTER_DIR)
        current_roster = normalize_roster(instance, initial_partial_roster)
    except Exception as e:
        print(f"Failed to load inputs: {e}")
        return

    initial_temp_dir = os.path.join(TMP_DIR, "initial_eval")

    try:
        if os.path.exists(INITIAL_VALIDATOR_OUTPUT_PATH):
            initial_raw = load_text_if_exists(INITIAL_VALIDATOR_OUTPUT_PATH)
            initial_result = parse_validator_output(initial_raw)
        else:
            initial_raw, initial_result = evaluate_roster_with_validator(
                instance,
                current_roster,
                initial_temp_dir,
            )
    except Exception as e:
        print(f"Validator failed on initial roster: {e}")
        return

    if not initial_result["hard_ok"]:
        print("Initial roster is infeasible according to INRC-II validator.")
        print(initial_result)
        return

    best_roster = deep_copy_roster(current_roster)
    best_raw = initial_raw
    best_result = initial_result

    current_result = initial_result
    current_raw = initial_raw

    validator_index_rows: List[Dict[str, Any]] = []
    accepted_changes: List[str] = []

    search_log = {
        "initial_total_cost": initial_result["total_cost"],
        "best_total_cost": initial_result["total_cost"],
        "iterations": [],
        "total_validator_calls": 0,
    }

    save_best_outputs(instance, initial_raw, best_raw, best_roster)

    print(f"Initial total cost: {initial_result['total_cost']}")
    print("Initial hard feasible:", initial_result["hard_ok"])

    for iteration in range(1, MAX_ITERS + 1):
        bundles = generate_deterministic_single_move_bundles(instance, current_roster)

        print(f"\nIteration {iteration}")
        print(f"Generated deterministic single-move bundles: {len(bundles)}")

        if not bundles:
            print("No deterministic bundles generated.")
            break

        old_cost = current_result["total_cost"]

        best_bundle, next_roster, next_raw, next_result = choose_best_bundle(
            instance=instance,
            current_roster=current_roster,
            current_result=current_result,
            bundles=bundles,
            iteration=iteration,
            validator_index_rows=validator_index_rows,
        )

        search_log["total_validator_calls"] = len(validator_index_rows)

        if best_bundle is None or next_roster is None or next_raw is None or next_result is None:
            print(f"Iteration {iteration}: no improving single move found.")

            search_log["iterations"].append({
                "iteration": iteration,
                "status": "no_improving_single_move_found",
                "accepted_bundle_id": None,
                "old_cost": old_cost,
                "new_cost": old_cost,
                "delta": 0,
                "changes": [],
            })

            break

        changes = diff_rosters(current_roster, next_roster)
        accepted_changes.extend(changes)

        current_roster = next_roster
        current_result = next_result
        current_raw = next_raw

        new_cost = next_result["total_cost"]
        delta = new_cost - old_cost

        print(
            f"Iteration {iteration}: accepted bundle {best_bundle.get('bundle_id')} "
            f"cost {old_cost} -> {new_cost} "
            f"delta {delta}"
        )

        for c in changes:
            print("  " + c)

        search_log["iterations"].append({
            "iteration": iteration,
            "status": "improved",
            "accepted_bundle_id": best_bundle.get("bundle_id", iteration),
            "accepted_bundle_reason": best_bundle.get("reason", ""),
            "old_cost": old_cost,
            "new_cost": new_cost,
            "delta": delta,
            "changes": changes,
            "accepted_bundle": best_bundle,
        })

        if current_result["total_cost"] < best_result["total_cost"]:
            best_roster = deep_copy_roster(current_roster)
            best_raw = current_raw
            best_result = current_result

            search_log["best_total_cost"] = best_result["total_cost"]

            save_best_outputs(instance, initial_raw, best_raw, best_roster)

    save_search_log(search_log)
    save_validator_calls_index(validator_index_rows)
    save_best_outputs(instance, initial_raw, best_raw, best_roster)

    print("\n==============================")
    print("Search finished")
    print("==============================")
    print(f"Initial cost: {initial_result['total_cost']}")
    print(f"Best cost:    {best_result['total_cost']}")
    print(f"Improvement:  {initial_result['total_cost'] - best_result['total_cost']}")
    print(f"Validator calls: {len(validator_index_rows)}")

    if accepted_changes:
        print("\nAccepted changes:")
        for line in accepted_changes:
            print(line)
    else:
        print("\nNo changes accepted.")

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


def main() -> None:
    random.seed(RANDOM_SEED)
    run_deterministic_improvement()


if __name__ == "__main__":
    main()