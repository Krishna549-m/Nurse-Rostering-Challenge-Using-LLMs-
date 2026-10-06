
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

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
MAX_ITERS = 2

MAX_CANDIDATES_PER_OPERATOR = 120
MAX_VALIDATOR_CALLS_PER_ITER = 80
MAX_REJECTED_STRATEGY_MEMORY = 12

PROMPT_TEMPLATE_PATH = "operator_strategy_prompt.txt"

INSTANCE_PATH = "response1.json"
INITIAL_ROSTER_DIR = "initial_roster"
INITIAL_VALIDATOR_OUTPUT_PATH = os.path.join(INITIAL_ROSTER_DIR, "validator_output.txt")

OUTPUT_DIR = "improved_roster_operator_agent"
TMP_DIR = "tmp_operator_candidates"
VALIDATOR_CALLS_DIR = os.path.join(OUTPUT_DIR, "validator_calls")
AGENT_CALLS_DIR = os.path.join(OUTPUT_DIR, "agent_calls")

VALIDATOR_DIR = "validator"
VALIDATOR_WEEKS_DIR = os.path.join(VALIDATOR_DIR, "weeks")
VALIDATOR_JAR = os.path.join(VALIDATOR_DIR, "validator.jar")
VALIDATOR_SCENARIO_FILE = os.path.join(VALIDATOR_DIR, "scenario.json")
VALIDATOR_HISTORY_FILE = os.path.join(VALIDATOR_DIR, "history.json")

API_KEY = ""
MODEL = "gpt-5.2"

DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_INDEX = {d: i for i, d in enumerate(DAY_ORDER)}

DAY_FULL_TO_SHORT = {
    "Monday": "Mon",
    "Tuesday": "Tue",
    "Wednesday": "Wed",
    "Thursday": "Thu",
    "Friday": "Fri",
    "Saturday": "Sat",
    "Sunday": "Sun",
}

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

SHIFT_FULL_TO_SHORT = {
    "Early": "E",
    "Late": "L",
    "Night": "N",
    "E": "E",
    "L": "L",
    "N": "N",
}

VALID_OPERATORS = {
    "reduce_overload",
    "repair_weekends",
    "repair_consecutive",
    "repair_optimal_coverage",
    "mixed_repair",
    "random_block_swap",
}

random.seed(RANDOM_SEED)


# ============================================================
# IO helpers
# ============================================================

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(VALIDATOR_CALLS_DIR, exist_ok=True)
    os.makedirs(AGENT_CALLS_DIR, exist_ok=True)


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


def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_day_label(day_label: str) -> Tuple[int, str]:
    m = re.match(r"W(\d+)_(Mon|Tue|Wed|Thu|Fri|Sat|Sun)$", day_label)
    if not m:
        raise ValueError(f"Invalid day label: {day_label}")
    return int(m.group(1)) - 1, m.group(2)


def day_sort_key(day: str) -> Tuple[int, int]:
    w, d = parse_day_label(day)
    return w, DAY_INDEX[d]


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
        raise FileNotFoundError("No initial_roster/week*.json files found")

    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]] = {}

    for week_file in week_files:
        with open(week_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        week_num = payload.get("week")
        if week_num is None:
            week_num = parse_week_from_filename(week_file) - 1

        for a in payload.get("assignments", []):
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
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]] = {}

    for nurse in instance["sets"]["N"]:
        roster[nurse] = {}
        for day in instance["sets"]["D"]:
            roster[nurse][day] = partial_roster.get(nurse, {}).get(day)

    return roster


# ============================================================
# Export roster
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
# Validator
# ============================================================

def parse_week_index_from_validator_week_file(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"week(\d+)\.json$", name)
    if not m:
        raise ValueError(f"Could not parse validator week index from: {path}")
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
        raise FileNotFoundError(f"No week*.json solution files in {solution_dir}")

    validator_week_files = get_validator_week_files()

    if len(sol_files) != len(validator_week_files):
        raise ValueError(
            f"Mismatch: solution week files={len(sol_files)}, "
            f"validator week files={len(validator_week_files)}"
        )

    return [
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
# Penalty extraction
# ============================================================

def day_index_to_label(instance: Dict[str, Any], day_index: Optional[int]) -> Optional[str]:
    if day_index is None:
        return None

    days = instance["sets"]["D"]
    if 0 <= day_index < len(days):
        return days[day_index]

    return None


def expand_day_window(
    instance: Dict[str, Any],
    center_day: str,
    radius: int = 3,
) -> List[str]:
    days = instance["sets"]["D"]
    if center_day not in days:
        return []

    idx = days.index(center_day)
    start = max(0, idx - radius)
    end = min(len(days), idx + radius + 1)

    return days[start:end]


def extract_penalty_atoms(
    instance: Dict[str, Any],
    validator_text: str,
) -> List[Dict[str, Any]]:
    atoms = []
    nurses = set(instance["sets"]["N"])

    current_nurse = None
    current_section = None

    section_map = {
        "Total assignments": "total_assignments",
        "Consecutive assignments": "consecutive_assignments",
        "Consecutive days off": "consecutive_days_off",
        "Weekend constraints": "weekend_constraints",
        "Preferences": "preferences",
    }

    lines = validator_text.splitlines()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        if line in nurses:
            current_nurse = line
            current_section = None
            continue

        if line in section_map:
            current_section = section_map[line]
            continue

        if line.startswith("-") and current_nurse is not None and current_section is not None:
            cost_match = re.search(r"cost:\s*(\d+)", line)
            day_match = re.search(r"day:\s*(\d+)", line)

            cost = int(cost_match.group(1)) if cost_match else 0
            day_index = int(day_match.group(1)) if day_match else None
            day_label = day_index_to_label(instance, day_index)

            atoms.append({
                "type": current_section,
                "nurse": current_nurse,
                "day_index": day_index,
                "day": day_label,
                "cost": cost,
                "text": line,
            })

    opt_pattern = re.compile(
        r"-\s*(\d+)\s+assignments short for optimal staffing "
        r"for shift type \(([^)]+)\) for skill \(([^)]+)\) "
        r"on day \(([^)]+)\) in week \((\d+)\), cost\s+(\d+)"
    )

    for raw_line in lines:
        line = raw_line.strip()
        m = opt_pattern.search(line)

        if not m:
            continue

        missing = int(m.group(1))
        shift = SHIFT_FULL_TO_SHORT.get(m.group(2), m.group(2))
        skill = m.group(3)
        day_full = m.group(4)
        week_zero = int(m.group(5))
        cost = int(m.group(6))

        day_short = DAY_FULL_TO_SHORT.get(day_full)
        day_label = week_day_to_label(week_zero, day_short) if day_short else None

        atoms.append({
            "type": "optimal_coverage",
            "nurse": None,
            "day_index": instance["sets"]["D"].index(day_label) if day_label in instance["sets"]["D"] else None,
            "day": day_label,
            "shift": shift,
            "skill": skill,
            "missing": missing,
            "cost": cost,
            "text": line,
        })

    atoms.sort(key=lambda x: x.get("cost", 0), reverse=True)
    return atoms


# ============================================================
# Workload and helper functions
# ============================================================

def compute_nurse_workload_stats(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> Dict[str, Dict[str, Any]]:
    stats = {}
    days = instance["sets"]["D"]

    for nurse in instance["sets"]["N"]:
        worked_days = [day for day in days if roster[nurse][day] is not None]

        weekend_count = 0
        weeks = sorted({parse_day_label(d)[0] for d in days})

        for week_zero in weeks:
            sat = week_day_to_label(week_zero, "Sat")
            sun = week_day_to_label(week_zero, "Sun")

            if roster[nurse][sat] is not None or roster[nurse][sun] is not None:
                weekend_count += 1

        contract = instance["parameters"]["contracts"][nurse]

        stats[nurse] = {
            "assignments": len(worked_days),
            "worked_days": worked_days,
            "weekends": weekend_count,
            "min_total": contract["min_total"],
            "max_total": contract["max_total"],
            "max_weekends": contract["max_weekends"],
            "over_max_total": max(0, len(worked_days) - contract["max_total"]),
            "under_min_total": max(0, contract["min_total"] - len(worked_days)),
            "over_max_weekends": max(0, weekend_count - contract["max_weekends"]),
        }

    return stats


def is_qualified(instance: Dict[str, Any], nurse: str, skill: str) -> bool:
    return instance["parameters"]["qualifications"][nurse].get(skill, 0) == 1


def off_qualified_nurses(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    day: str,
    skill: str,
) -> List[str]:
    stats = compute_nurse_workload_stats(instance, roster)

    candidates = []

    for nurse in instance["sets"]["N"]:
        if roster[nurse][day] is not None:
            continue
        if not is_qualified(instance, nurse, skill):
            continue
        candidates.append(nurse)

    candidates.sort(
        key=lambda n: (
            stats[n]["over_max_total"],
            stats[n]["over_max_weekends"],
            stats[n]["assignments"],
        )
    )

    return candidates


def nurse_has_weekend(
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    nurse: str,
    week_zero: int,
) -> bool:
    sat = week_day_to_label(week_zero, "Sat")
    sun = week_day_to_label(week_zero, "Sun")
    return roster[nurse][sat] is not None or roster[nurse][sun] is not None


def weekend_days_for_all_weeks(instance: Dict[str, Any]) -> List[Tuple[int, str, str]]:
    weeks = sorted({parse_day_label(d)[0] for d in instance["sets"]["D"]})
    return [(w, week_day_to_label(w, "Sat"), week_day_to_label(w, "Sun")) for w in weeks]


def get_top_nurses_from_atoms(
    atoms: List[Dict[str, Any]],
    types: Optional[set] = None,
    limit: int = 5,
) -> List[str]:
    scores: Dict[str, int] = {}

    for atom in atoms:
        if types is not None and atom["type"] not in types:
            continue

        nurse = atom.get("nurse")
        if nurse is None:
            continue

        scores[nurse] = scores.get(nurse, 0) + int(atom.get("cost", 0))

    return [n for n, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]]


# ============================================================
# Move and bundle application
# ============================================================

def move_signature(move: Dict[str, Any]) -> str:
    return json.dumps(move, sort_keys=True)


def bundle_signature(bundle: Dict[str, Any]) -> str:
    return json.dumps(bundle.get("moves", []), sort_keys=True)


def quick_move_precheck(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    move: Dict[str, Any],
) -> bool:
    nurses = set(instance["sets"]["N"])
    days = set(instance["sets"]["D"])
    shifts = set(instance["sets"]["S"])
    skills = set(instance["sets"]["K"])

    try:
        t = move["type"]

        if t == "reassign":
            day = move["day"]
            src = move["from_nurse"]
            dst = move["to_nurse"]
            shift = move["shift"]
            skill = move["skill"]

            return (
                day in days
                and src in nurses
                and dst in nurses
                and src != dst
                and shift in shifts
                and skill in skills
                and roster[src][day] == (shift, skill)
                and roster[dst][day] is None
                and is_qualified(instance, dst, skill)
            )

        if t == "swap":
            day = move["day"]
            a = move["nurse_a"]
            b = move["nurse_b"]

            if day not in days or a not in nurses or b not in nurses or a == b:
                return False

            assn_a = roster[a][day]
            assn_b = roster[b][day]

            if assn_a is None and assn_b is None:
                return False

            if assn_a is not None and not is_qualified(instance, b, assn_a[1]):
                return False

            if assn_b is not None and not is_qualified(instance, a, assn_b[1]):
                return False

            return True

        if t == "coverage_fill":
            day = move["day"]
            shift = move["shift"]
            skill = move["skill"]
            nurse = move["to_nurse"]

            return (
                day in days
                and shift in shifts
                and skill in skills
                and nurse in nurses
                and roster[nurse][day] is None
                and is_qualified(instance, nurse, skill)
            )

        if t == "block_swap":
            a = move["nurse_a"]
            b = move["nurse_b"]
            block_days = move["days"]

            if a not in nurses or b not in nurses or a == b:
                return False

            if not isinstance(block_days, list) or not block_days:
                return False

            any_change = False

            for day in block_days:
                if day not in days:
                    return False

                assn_a = roster[a][day]
                assn_b = roster[b][day]

                if assn_a is not None or assn_b is not None:
                    any_change = True

                if assn_a is not None and not is_qualified(instance, b, assn_a[1]):
                    return False

                if assn_b is not None and not is_qualified(instance, a, assn_b[1]):
                    return False

            return any_change

        if t == "block_reassign":
            src = move["from_nurse"]
            dst = move["to_nurse"]
            block_days = move["days"]

            if src not in nurses or dst not in nurses or src == dst:
                return False

            if not isinstance(block_days, list) or not block_days:
                return False

            any_change = False

            for day in block_days:
                if day not in days:
                    return False

                assn = roster[src][day]
                if assn is None:
                    continue

                any_change = True

                if roster[dst][day] is not None:
                    return False

                if not is_qualified(instance, dst, assn[1]):
                    return False

            return any_change

    except Exception:
        return False

    return False


def apply_move(
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    move: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]]:
    trial = deep_copy_roster(roster)

    try:
        t = move["type"]

        if t == "reassign":
            day = move["day"]
            src = move["from_nurse"]
            dst = move["to_nurse"]
            trial[dst][day] = trial[src][day]
            trial[src][day] = None
            return trial

        if t == "swap":
            day = move["day"]
            a = move["nurse_a"]
            b = move["nurse_b"]
            trial[a][day], trial[b][day] = trial[b][day], trial[a][day]
            return trial

        if t == "coverage_fill":
            day = move["day"]
            shift = move["shift"]
            skill = move["skill"]
            nurse = move["to_nurse"]
            trial[nurse][day] = (shift, skill)
            return trial

        if t == "block_swap":
            a = move["nurse_a"]
            b = move["nurse_b"]
            for day in move["days"]:
                trial[a][day], trial[b][day] = trial[b][day], trial[a][day]
            return trial

        if t == "block_reassign":
            src = move["from_nurse"]
            dst = move["to_nurse"]
            for day in move["days"]:
                if trial[src][day] is None:
                    continue
                trial[dst][day] = trial[src][day]
                trial[src][day] = None
            return trial

    except Exception:
        return None

    return None


def apply_bundle(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    bundle: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]]:
    moves = bundle.get("moves", [])
    if not isinstance(moves, list) or not moves:
        return None

    trial = deep_copy_roster(roster)

    for move in moves:
        if not quick_move_precheck(instance, trial, move):
            return None

        next_trial = apply_move(trial, move)
        if next_trial is None:
            return None

        trial = next_trial

    return trial


# ============================================================
# Deterministic candidate-generating operators
# ============================================================

def make_bundle(operator: str, reason: str, moves: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "operator": operator,
        "reason": reason,
        "moves": moves,
        "meta": meta or {},
    }


def generate_reduce_overload_candidates(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    stats = compute_nurse_workload_stats(instance, roster)
    atoms = extract_penalty_atoms(instance, validator_text)

    target_nurses = strategy.get("target_nurses") or []
    target_nurses = [n for n in target_nurses if n in instance["sets"]["N"]]

    if not target_nurses:
        target_nurses = [
            n for n, s in sorted(
                stats.items(),
                key=lambda kv: (kv[1]["over_max_total"], kv[1]["assignments"]),
                reverse=True,
            )
            if s["over_max_total"] > 0
        ]

    if not target_nurses:
        target_nurses = get_top_nurses_from_atoms(atoms, {"total_assignments"}, limit=5)

    candidates = []
    seen = set()

    for src in target_nurses:
        worked_days = list(stats[src]["worked_days"])

        target_days = strategy.get("target_days") or []
        target_days = [d for d in target_days if d in worked_days]

        days_to_try = target_days if target_days else worked_days
        days_to_try = sorted(days_to_try, key=day_sort_key)

        helpers = [
            n for n in instance["sets"]["N"]
            if n != src and stats[n]["over_max_total"] == 0
        ]

        helpers.sort(key=lambda n: (stats[n]["under_min_total"] == 0, stats[n]["assignments"], stats[n]["weekends"]))

        for day in days_to_try:
            assn = roster[src][day]
            if assn is None:
                continue

            shift, skill = assn

            for dst in helpers:
                move = {
                    "type": "reassign",
                    "day": day,
                    "from_nurse": src,
                    "to_nurse": dst,
                    "shift": shift,
                    "skill": skill,
                }

                bundle = make_bundle(
                    "reduce_overload",
                    f"Move {src}'s {day} {shift}/{skill} assignment to underloaded qualified nurse {dst}.",
                    [move],
                    {"source": src, "target": dst, "day": day},
                )

                sig = bundle_signature(bundle)

                if sig in seen:
                    continue

                seen.add(sig)
                candidates.append(bundle)

                if len(candidates) >= MAX_CANDIDATES_PER_OPERATOR:
                    return candidates

    return candidates


def generate_repair_weekends_candidates(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    stats = compute_nurse_workload_stats(instance, roster)
    atoms = extract_penalty_atoms(instance, validator_text)

    target_nurses = strategy.get("target_nurses") or []
    target_nurses = [n for n in target_nurses if n in instance["sets"]["N"]]

    if not target_nurses:
        target_nurses = [
            n for n, s in sorted(
                stats.items(),
                key=lambda kv: (kv[1]["over_max_weekends"], kv[1]["weekends"]),
                reverse=True,
            )
            if s["over_max_weekends"] > 0
        ]

    if not target_nurses:
        target_nurses = get_top_nurses_from_atoms(atoms, {"weekend_constraints"}, limit=5)

    candidates = []
    seen = set()

    for src in target_nurses:
        helpers = [
            n for n in instance["sets"]["N"]
            if n != src and stats[n]["over_max_weekends"] == 0 and stats[n]["over_max_total"] == 0
        ]
        helpers.sort(key=lambda n: (stats[n]["weekends"], stats[n]["assignments"]))

        for week_zero, sat, sun in weekend_days_for_all_weeks(instance):
            src_works_weekend = roster[src][sat] is not None or roster[src][sun] is not None
            if not src_works_weekend:
                continue

            block_days = [sat, sun]

            for dst in helpers:
                # Try block reassign if dst is off on both days and qualified for assignments.
                can_block_reassign = True
                for day in block_days:
                    assn = roster[src][day]
                    if assn is None:
                        continue
                    if roster[dst][day] is not None or not is_qualified(instance, dst, assn[1]):
                        can_block_reassign = False
                        break

                if can_block_reassign:
                    bundle = make_bundle(
                        "repair_weekends",
                        f"Move weekend block from {src} to {dst} on week {week_zero + 1}.",
                        [{
                            "type": "block_reassign",
                            "from_nurse": src,
                            "to_nurse": dst,
                            "days": block_days,
                        }],
                        {"source": src, "target": dst, "week": week_zero + 1},
                    )

                    sig = bundle_signature(bundle)
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(bundle)

                # Try block swap even if dst works one of the days.
                bundle = make_bundle(
                    "repair_weekends",
                    f"Swap weekend block between {src} and {dst} on week {week_zero + 1}.",
                    [{
                        "type": "block_swap",
                        "nurse_a": src,
                        "nurse_b": dst,
                        "days": block_days,
                    }],
                    {"source": src, "target": dst, "week": week_zero + 1},
                )

                sig = bundle_signature(bundle)
                if sig not in seen:
                    seen.add(sig)
                    candidates.append(bundle)

                if len(candidates) >= MAX_CANDIDATES_PER_OPERATOR:
                    return candidates

    return candidates


def generate_repair_consecutive_candidates(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    stats = compute_nurse_workload_stats(instance, roster)
    atoms = extract_penalty_atoms(instance, validator_text)

    relevant_atoms = [a for a in atoms if a["type"] in {"consecutive_assignments", "consecutive_days_off"}]

    if strategy.get("target_nurses"):
        relevant_atoms = [a for a in relevant_atoms if a.get("nurse") in strategy["target_nurses"]]

    candidates = []
    seen = set()

    for atom in relevant_atoms:
        src = atom.get("nurse")
        center_day = atom.get("day")

        if not src or not center_day:
            continue

        window = expand_day_window(instance, center_day, radius=2)

        helpers = [
            n for n in instance["sets"]["N"]
            if n != src and stats[n]["over_max_total"] == 0
        ]
        helpers.sort(key=lambda n: (stats[n]["assignments"], stats[n]["weekends"]))

        for dst in helpers[:12]:
            # Try 2-day and 3-day block swaps around the atom.
            for block_len in [2, 3, 4]:
                for start in range(0, max(0, len(window) - block_len + 1)):
                    block_days = window[start:start + block_len]

                    bundle = make_bundle(
                        "repair_consecutive",
                        f"Try block swap around consecutive penalty for {src} with {dst}.",
                        [{
                            "type": "block_swap",
                            "nurse_a": src,
                            "nurse_b": dst,
                            "days": block_days,
                        }],
                        {"source": src, "target": dst, "atom": atom},
                    )

                    sig = bundle_signature(bundle)
                    if sig not in seen:
                        seen.add(sig)
                        candidates.append(bundle)

                    # Try block reassign only if destination is off and qualified for all src assignments.
                    can_reassign = True
                    moved_any = False
                    for day in block_days:
                        assn = roster[src][day]
                        if assn is None:
                            continue
                        moved_any = True
                        if roster[dst][day] is not None or not is_qualified(instance, dst, assn[1]):
                            can_reassign = False
                            break

                    if can_reassign and moved_any:
                        bundle = make_bundle(
                            "repair_consecutive",
                            f"Try block reassign around consecutive penalty from {src} to {dst}.",
                            [{
                                "type": "block_reassign",
                                "from_nurse": src,
                                "to_nurse": dst,
                                "days": block_days,
                            }],
                            {"source": src, "target": dst, "atom": atom},
                        )

                        sig = bundle_signature(bundle)
                        if sig not in seen:
                            seen.add(sig)
                            candidates.append(bundle)

                    if len(candidates) >= MAX_CANDIDATES_PER_OPERATOR:
                        return candidates

    return candidates


def generate_repair_optimal_coverage_candidates(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    atoms = extract_penalty_atoms(instance, validator_text)
    stats = compute_nurse_workload_stats(instance, roster)

    opt_atoms = [a for a in atoms if a["type"] == "optimal_coverage"]

    candidates = []
    seen = set()

    for atom in opt_atoms:
        day = atom.get("day")
        shift = atom.get("shift")
        skill = atom.get("skill")

        if not day or not shift or not skill:
            continue

        helpers = off_qualified_nurses(instance, roster, day, skill)
        helpers.sort(key=lambda n: (stats[n]["over_max_total"], stats[n]["over_max_weekends"], stats[n]["assignments"]))

        for nurse in helpers[:20]:
            bundle = make_bundle(
                "repair_optimal_coverage",
                f"Fill optimal coverage shortage on {day} {shift}/{skill} with {nurse}.",
                [{
                    "type": "coverage_fill",
                    "day": day,
                    "shift": shift,
                    "skill": skill,
                    "to_nurse": nurse,
                }],
                {"target_day": day, "shift": shift, "skill": skill, "nurse": nurse},
            )

            sig = bundle_signature(bundle)
            if sig in seen:
                continue

            seen.add(sig)
            candidates.append(bundle)

            if len(candidates) >= MAX_CANDIDATES_PER_OPERATOR:
                return candidates

    return candidates


def generate_random_block_swap_candidates(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    nurses = list(instance["sets"]["N"])
    days = sorted(instance["sets"]["D"], key=day_sort_key)

    candidates = []
    seen = set()

    for _ in range(MAX_CANDIDATES_PER_OPERATOR * 4):
        a, b = random.sample(nurses, 2)

        start = random.randint(0, max(0, len(days) - 4))
        block_len = random.choice([2, 3, 4])
        block_days = days[start:start + block_len]

        bundle = make_bundle(
            "random_block_swap",
            f"Random block swap between {a} and {b}.",
            [{
                "type": "block_swap",
                "nurse_a": a,
                "nurse_b": b,
                "days": block_days,
            }],
            {"nurse_a": a, "nurse_b": b, "days": block_days},
        )

        sig = bundle_signature(bundle)
        if sig in seen:
            continue

        seen.add(sig)
        candidates.append(bundle)

        if len(candidates) >= MAX_CANDIDATES_PER_OPERATOR:
            return candidates

    return candidates


def generate_candidates_for_operator(
    operator: str,
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    strategy: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if operator == "reduce_overload":
        return generate_reduce_overload_candidates(instance, roster, validator_text, strategy)

    if operator == "repair_weekends":
        return generate_repair_weekends_candidates(instance, roster, validator_text, strategy)

    if operator == "repair_consecutive":
        return generate_repair_consecutive_candidates(instance, roster, validator_text, strategy)

    if operator == "repair_optimal_coverage":
        return generate_repair_optimal_coverage_candidates(instance, roster, validator_text, strategy)

    if operator == "random_block_swap":
        return generate_random_block_swap_candidates(instance, roster, validator_text, strategy)

    if operator == "mixed_repair":
        combined = []
        for op in ["reduce_overload", "repair_weekends", "repair_consecutive", "repair_optimal_coverage", "random_block_swap"]:
            combined.extend(generate_candidates_for_operator(op, instance, roster, validator_text, {**strategy, "operator": op}))
            if len(combined) >= MAX_CANDIDATES_PER_OPERATOR:
                break
        return combined[:MAX_CANDIDATES_PER_OPERATOR]

    return []


# ============================================================
# Agent strategy selection
# ============================================================

def compact_cost_summary(parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "total_cost": parsed["total_cost"],
        "hard_ok": parsed["hard_ok"],
        "total_assignment_cost": parsed["total_assignment_cost"],
        "consecutive_cost": parsed["consecutive_cost"],
        "non_working_days_cost": parsed["non_working_days_cost"],
        "preferences_cost": parsed["preferences_cost"],
        "max_working_weekend_cost": parsed["max_working_weekend_cost"],
        "complete_weekends_cost": parsed["complete_weekends_cost"],
        "optimal_coverage_cost": parsed["optimal_coverage_cost"],
    }


def heuristic_strategy(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    parsed: Dict[str, Any],
    strategy_memory: List[Dict[str, Any]],
) -> Dict[str, Any]:
    atoms = extract_penalty_atoms(instance, validator_text)
    stats = compute_nurse_workload_stats(instance, roster)

    scores = {
        "reduce_overload": parsed["total_assignment_cost"],
        "repair_weekends": parsed["max_working_weekend_cost"],
        "repair_consecutive": parsed["consecutive_cost"] + parsed["non_working_days_cost"],
        "repair_optimal_coverage": parsed["optimal_coverage_cost"],
        "mixed_repair": 0,
        "random_block_swap": 0,
    }

    recent_failed_ops = [
        m.get("operator") for m in strategy_memory[-5:]
        if m.get("status") == "no_improvement"
    ]

    for op in recent_failed_ops:
        if op in scores:
            scores[op] = int(scores[op] * 0.75)

    if len(strategy_memory) >= 5 and all(m.get("status") == "no_improvement" for m in strategy_memory[-5:]):
        operator = "mixed_repair"
    elif len(strategy_memory) >= 8 and all(m.get("status") == "no_improvement" for m in strategy_memory[-8:]):
        operator = "random_block_swap"
    else:
        operator = max(scores.items(), key=lambda kv: kv[1])[0]

    target_nurses = []

    if operator == "reduce_overload":
        target_nurses = [
            n for n, s in sorted(stats.items(), key=lambda kv: kv[1]["over_max_total"], reverse=True)
            if s["over_max_total"] > 0
        ][:5]

    elif operator == "repair_weekends":
        target_nurses = [
            n for n, s in sorted(stats.items(), key=lambda kv: kv[1]["over_max_weekends"], reverse=True)
            if s["over_max_weekends"] > 0
        ][:5]

    elif operator == "repair_consecutive":
        target_nurses = get_top_nurses_from_atoms(
            atoms,
            {"consecutive_assignments", "consecutive_days_off"},
            limit=5,
        )

    return {
        "operator": operator,
        "target_nurses": target_nurses,
        "target_days": [],
        "reason": "Heuristic strategy selected from current largest penalty categories.",
        "avoid": [],
    }


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


def build_strategy_prompt(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    parsed: Dict[str, Any],
    strategy_memory: List[Dict[str, Any]],
) -> str:
    template = load_prompt_template(PROMPT_TEMPLATE_PATH)

    atoms = extract_penalty_atoms(instance, validator_text)
    stats = compute_nurse_workload_stats(instance, roster)

    payload = {
        "current_cost_summary": compact_cost_summary(parsed),
        "top_penalty_atoms": atoms[:40],
        "workload_stats": stats,
        "previous_strategy_outcomes": strategy_memory[-MAX_REJECTED_STRATEGY_MEMORY:],
        "available_operators": sorted(VALID_OPERATORS),
        "valid_nurses": instance["sets"]["N"],
        "valid_days": instance["sets"]["D"],
    }

    return template.replace("{{PAYLOAD_JSON}}", json.dumps(payload, indent=2))


def call_strategy_agent(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    parsed: Dict[str, Any],
    strategy_memory: List[Dict[str, Any]],
    iteration: int,
) -> Dict[str, Any]:
    prompt = build_strategy_prompt(instance, roster, validator_text, parsed, strategy_memory)

    raw = ""

    if API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=API_KEY)
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                temperature=0.1,
                max_output_tokens=1500,
                stream=False,
            )
            raw = response.output_text or ""
            strategy = extract_json_object(raw)
        except Exception as e:
            raw = f"AGENT_ERROR: {e}"
            strategy = None
    else:
        strategy = None
        raw = "NO_API_KEY_OR_OPENAI_PACKAGE: using heuristic strategy."

    if not strategy:
        strategy = heuristic_strategy(instance, roster, validator_text, parsed, strategy_memory)

    operator = strategy.get("operator")
    if operator not in VALID_OPERATORS:
        strategy = heuristic_strategy(instance, roster, validator_text, parsed, strategy_memory)

    strategy["target_nurses"] = [
        n for n in strategy.get("target_nurses", [])
        if n in instance["sets"]["N"]
    ]

    strategy["target_days"] = [
        d for d in strategy.get("target_days", [])
        if d in instance["sets"]["D"]
    ]

    save_agent_call(iteration, prompt, raw, strategy)
    return strategy


def save_agent_call(iteration: int, prompt: str, raw: str, strategy: Dict[str, Any]) -> None:
    base = f"iter_{iteration:04d}"

    with open(os.path.join(AGENT_CALLS_DIR, f"{base}_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    with open(os.path.join(AGENT_CALLS_DIR, f"{base}_raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    with open(os.path.join(AGENT_CALLS_DIR, f"{base}_strategy.json"), "w", encoding="utf-8") as f:
        json.dump(strategy, f, indent=2)


# ============================================================
# Candidate evaluation
# ============================================================

def save_validator_call(
    iteration: int,
    candidate_index: int,
    raw_text: str,
    parsed: Dict[str, Any],
    bundle: Dict[str, Any],
) -> None:
    base = f"iter_{iteration:04d}_candidate_{candidate_index:03d}"

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}_parsed.json"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}_bundle.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)


def bundle_sort_key(result: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    return (
        result["total_cost"],
        result["non_working_days_cost"],
        result["optimal_coverage_cost"],
        result["consecutive_cost"],
        result["total_assignment_cost"],
    )


def diff_rosters(
    before: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    after: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
) -> List[str]:
    changes = []

    for nurse in sorted(before.keys()):
        for day in sorted(before[nurse].keys(), key=day_sort_key):
            old = before[nurse][day]
            new = after[nurse][day]

            if old != new:
                old_str = "OFF" if old is None else f"{old[0]}/{old[1]}"
                new_str = "OFF" if new is None else f"{new[0]}/{new[1]}"
                changes.append(f"{day}: {nurse} {old_str} -> {new_str}")

    return changes


def evaluate_candidates(
    instance: Dict[str, Any],
    current_roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    current_result: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    iteration: int,
    validator_rows: List[Dict[str, Any]],
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]],
    Optional[str],
    Optional[Dict[str, Any]],
]:
    old_cost = current_result["total_cost"]

    best_bundle = None
    best_roster = None
    best_raw = None
    best_result = None

    seen = set()
    evaluated = 0

    for candidate_index, bundle in enumerate(candidates, start=1):
        sig = bundle_signature(bundle)
        if sig in seen:
            continue
        seen.add(sig)

        trial_roster = apply_bundle(instance, current_roster, bundle)
        if trial_roster is None:
            continue

        evaluated += 1
        if evaluated > MAX_VALIDATOR_CALLS_PER_ITER:
            break

        candidate_dir = os.path.join(TMP_DIR, f"iter_{iteration:04d}_candidate_{candidate_index:03d}")

        try:
            raw, parsed = evaluate_roster_with_validator(instance, trial_roster, candidate_dir)
        except Exception as e:
            continue

        save_validator_call(iteration, candidate_index, raw, parsed, bundle)

        delta = parsed["total_cost"] - old_cost

        validator_rows.append({
            "iteration": iteration,
            "candidate_index": candidate_index,
            "operator": bundle.get("operator", ""),
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
            "bundle_signature": sig,
        })

        if not parsed["hard_ok"]:
            continue

        if parsed["total_cost"] < old_cost:
            if best_result is None or bundle_sort_key(parsed) < bundle_sort_key(best_result):
                best_bundle = bundle
                best_roster = trial_roster
                best_raw = raw
                best_result = parsed

    return best_bundle, best_roster, best_raw, best_result


# ============================================================
# Save outputs
# ============================================================

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


def save_search_log(log_data: Dict[str, Any]) -> None:
    with open(os.path.join(OUTPUT_DIR, "search_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


def save_strategy_memory(strategy_memory: List[Dict[str, Any]]) -> None:
    with open(os.path.join(OUTPUT_DIR, "strategy_memory.json"), "w", encoding="utf-8") as f:
        json.dump(strategy_memory, f, indent=2)


def save_validator_rows(rows: List[Dict[str, Any]]) -> None:
    path = os.path.join(OUTPUT_DIR, "validator_calls_index.csv")

    fieldnames = [
        "iteration",
        "candidate_index",
        "operator",
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
        "bundle_signature",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Main loop
# ============================================================

def run_operator_agent_search() -> None:
    ensure_dirs()

    if not os.path.exists(INSTANCE_PATH):
        print(f"Missing instance file: {INSTANCE_PATH}")
        return

    if not os.path.exists(PROMPT_TEMPLATE_PATH):
        print(f"Missing prompt template: {PROMPT_TEMPLATE_PATH}")
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

    current_raw = initial_raw
    current_result = initial_result

    strategy_memory: List[Dict[str, Any]] = []
    validator_rows: List[Dict[str, Any]] = []

    search_log = {
        "method": "agentic_operator_selection_with_deterministic_candidate_generation",
        "initial_total_cost": initial_result["total_cost"],
        "best_total_cost": initial_result["total_cost"],
        "iterations": [],
    }

    save_best_outputs(instance, initial_raw, best_raw, best_roster)

    print("Initial hard feasible:", initial_result["hard_ok"])
    print("Initial total cost:", initial_result["total_cost"])

    no_improvement_streak = 0

    for iteration in range(1, MAX_ITERS + 1):
        print(f"\nIteration {iteration}")
        print(f"Current cost: {current_result['total_cost']}")

        strategy = call_strategy_agent(
            instance=instance,
            roster=current_roster,
            validator_text=current_raw,
            parsed=current_result,
            strategy_memory=strategy_memory,
            iteration=iteration,
        )

        operator = strategy["operator"]
        print(f"Selected operator: {operator}")
        print(f"Reason: {strategy.get('reason', '')}")

        candidates = generate_candidates_for_operator(
            operator=operator,
            instance=instance,
            roster=current_roster,
            validator_text=current_raw,
            strategy=strategy,
        )

        print(f"Generated candidates: {len(candidates)}")

        if not candidates:
            memory_record = {
                "iteration": iteration,
                "operator": operator,
                "status": "no_candidates",
                "strategy": strategy,
                "old_cost": current_result["total_cost"],
                "new_cost": current_result["total_cost"],
                "delta": 0,
            }
            strategy_memory.append(memory_record)
            search_log["iterations"].append(memory_record)
            save_strategy_memory(strategy_memory)
            save_search_log(search_log)
            no_improvement_streak += 1
            continue

        old_cost = current_result["total_cost"]

        best_bundle, next_roster, next_raw, next_result = evaluate_candidates(
            instance=instance,
            current_roster=current_roster,
            current_result=current_result,
            candidates=candidates,
            iteration=iteration,
            validator_rows=validator_rows,
        )

        if best_bundle is None:
            print("No improving hard-feasible candidate found.")

            memory_record = {
                "iteration": iteration,
                "operator": operator,
                "status": "no_improvement",
                "strategy": strategy,
                "candidate_count": len(candidates),
                "old_cost": old_cost,
                "new_cost": old_cost,
                "delta": 0,
            }

            strategy_memory.append(memory_record)
            search_log["iterations"].append(memory_record)
            no_improvement_streak += 1

        else:
            changes = diff_rosters(current_roster, next_roster)
            new_cost = next_result["total_cost"]
            delta = new_cost - old_cost

            print(f"Accepted improvement: {old_cost} -> {new_cost}, delta {delta}")
            for c in changes:
                print("  " + c)

            current_roster = next_roster
            current_raw = next_raw
            current_result = next_result

            memory_record = {
                "iteration": iteration,
                "operator": operator,
                "status": "improved",
                "strategy": strategy,
                "candidate_count": len(candidates),
                "old_cost": old_cost,
                "new_cost": new_cost,
                "delta": delta,
                "accepted_bundle": best_bundle,
                "changes": changes,
            }

            strategy_memory.append(memory_record)
            search_log["iterations"].append(memory_record)
            no_improvement_streak = 0

            if current_result["total_cost"] < best_result["total_cost"]:
                best_roster = deep_copy_roster(current_roster)
                best_raw = current_raw
                best_result = current_result
                search_log["best_total_cost"] = best_result["total_cost"]
                save_best_outputs(instance, initial_raw, best_raw, best_roster)

        save_strategy_memory(strategy_memory)
        save_search_log(search_log)
        save_validator_rows(validator_rows)

        if no_improvement_streak >= 10:
            print("Stopping: 10 consecutive iterations without improvement.")
            break

    save_strategy_memory(strategy_memory)
    save_search_log(search_log)
    save_validator_rows(validator_rows)
    save_best_outputs(instance, initial_raw, best_raw, best_roster)

    print("\n==============================")
    print("Search finished")
    print("==============================")
    print(f"Initial cost: {initial_result['total_cost']}")
    print(f"Best cost:    {best_result['total_cost']}")
    print(f"Improvement:  {initial_result['total_cost'] - best_result['total_cost']}")
    print(f"Validator calls: {len(validator_rows)}")
    print(f"Outputs saved in: {OUTPUT_DIR}")


def main() -> None:
    random.seed(RANDOM_SEED)
    run_operator_agent_search()


if __name__ == "__main__":
    main()
