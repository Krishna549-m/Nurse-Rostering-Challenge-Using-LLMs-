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

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
MAX_ITERS = 2

MAX_AGENT_BUNDLES = 20
MAX_MOVES_PER_BUNDLE = 3
MAX_NEIGHBOURHOODS = 8
MAX_DAYS_PER_NEIGHBOURHOOD = 10
MAX_NURSES_PER_NEIGHBOURHOOD = 5
MAX_REJECTED_MEMORY = 10

AGENT_PROMPT_TEMPLATE_PATH = "agentic_repair_prompt.txt"

INSTANCE_PATH = "response1.json"

INITIAL_ROSTER_DIR = "initial_roster"
INITIAL_VALIDATOR_OUTPUT_PATH = os.path.join(INITIAL_ROSTER_DIR, "validator_output.txt")

OUTPUT_DIR = "improved_roster_agentic_block"
TMP_DIR = "tmp_candidates_agentic_block"
VALIDATOR_CALLS_DIR = os.path.join(OUTPUT_DIR, "validator_calls")
LLM_CALLS_DIR = os.path.join(OUTPUT_DIR, "llm_calls")

VALIDATOR_DIR = "validator"
VALIDATOR_WEEKS_DIR = os.path.join(VALIDATOR_DIR, "weeks")
VALIDATOR_JAR = os.path.join(VALIDATOR_DIR, "validator.jar")
VALIDATOR_SCENARIO_FILE = os.path.join(VALIDATOR_DIR, "scenario.json")
VALIDATOR_HISTORY_FILE = os.path.join(VALIDATOR_DIR, "history.json")

API_KEY =  ""
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

random.seed(RANDOM_SEED)


# ============================================================
# Basic helpers
# ============================================================

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(VALIDATOR_CALLS_DIR, exist_ok=True)
    os.makedirs(LLM_CALLS_DIR, exist_ok=True)


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
# Penalty atom extraction
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
# Workload, hard context, neighbourhoods
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


def build_hard_constraint_context(instance: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hard_constraints": {
            "H1_single_assignment_per_day": {
                "description": "A nurse can work at most one shift on the same day.",
                "avoid": [
                    "Do not assign a nurse to a day where they already work.",
                    "Do not use coverage_fill if the target nurse is already assigned on that day.",
                    "Do not block_reassign into a nurse if that nurse already works on any selected day.",
                ],
            },
            "H2_minimum_coverage": {
                "description": "For every day, shift, and skill, assignments must be at least the minimum required coverage.",
                "avoid": [
                    "Do not remove assignments unless minimum coverage is likely still satisfied.",
                    "Do not move a nurse away from a day/shift/skill if that assignment may be required for minimum coverage.",
                    "When using remove_block, prefer removing assignments only from overstaffed shift-skill-day combinations.",
                ],
            },
            "H3_required_skill": {
                "description": "A nurse assigned to a shift-skill must be qualified for that skill.",
                "avoid": [
                    "Only assign a nurse to a skill if qualifications[nurse][skill] == 1."
                ],
            },
            "H4_forbidden_shift_successions": {
                "description": "Forbidden shift transitions cannot occur across consecutive days.",
                "forbidden_successions": instance["parameters"].get("shift_successions", {}).get("forbidden", []),
                "avoid": [
                    "Do not create Late -> Early on consecutive days.",
                    "Do not create Night -> Early on consecutive days.",
                    "Do not create Night -> Late on consecutive days.",
                    "Check the day before and day after each changed day.",
                ],
            },
            "H5_historical_boundary_conditions": {
                "description": "The first days of the planning horizon must respect historical last shift and consecutive work/off data.",
                "historical_assignments": instance["parameters"].get("historical_assignments", {}),
                "avoid": [
                    "Be extra careful when changing W1_Mon, W1_Tue, or W1_Wed."
                ],
            },
        },
        "validator_note": (
            "The INRC-II validator is the final authority. "
            "The agent should avoid obviously invalid moves to reduce wasted validator calls."
        ),
    }


def get_off_qualified_nurses(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    day: str,
    skill: str,
) -> List[str]:
    result = []
    qualifications = instance["parameters"]["qualifications"]

    for nurse in instance["sets"]["N"]:
        if roster[nurse][day] is not None:
            continue

        if qualifications[nurse].get(skill, 0) != 1:
            continue

        result.append(nurse)

    return result


def generate_block_repair_neighbourhoods(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    max_neighbourhoods: int = MAX_NEIGHBOURHOODS,
) -> List[Dict[str, Any]]:
    atoms = extract_penalty_atoms(instance, validator_text)
    workload_stats = compute_nurse_workload_stats(instance, roster)

    neighbourhoods = []
    seen = set()

    for atom in atoms:
        if len(neighbourhoods) >= max_neighbourhoods:
            break

        target_type = atom["type"]
        target_nurse = atom.get("nurse")
        cost = atom.get("cost", 0)

        if target_nurse is None:
            continue

        if target_type == "preferences":
            continue

        center_day = atom.get("day")

        if center_day:
            days = expand_day_window(instance, center_day, radius=3)
        else:
            worked = workload_stats[target_nurse]["worked_days"]
            days = worked[:MAX_DAYS_PER_NEIGHBOURHOOD]

        if not days:
            continue

        helper_nurses = [n for n in instance["sets"]["N"] if n != target_nurse]
        helper_nurses.sort(
            key=lambda n: (
                workload_stats[n]["over_max_total"],
                workload_stats[n]["over_max_weekends"],
                workload_stats[n]["assignments"],
            )
        )

        selected_nurses = [target_nurse] + helper_nurses[:MAX_NURSES_PER_NEIGHBOURHOOD - 1]

        key = (target_type, tuple(selected_nurses), tuple(days))

        if key in seen:
            continue

        seen.add(key)

        neighbourhoods.append({
            "neighbourhood_id": len(neighbourhoods) + 1,
            "target_penalty": target_type,
            "target_cost": cost,
            "target_nurses": [target_nurse],
            "nurses": selected_nurses,
            "days": days[:MAX_DAYS_PER_NEIGHBOURHOOD],
            "allowed_move_types": [
                "block_reassign",
                "block_swap",
                "remove_block",
                "reassign",
                "swap",
            ],
            "reason": atom["text"],
        })

    for atom in atoms:
        if len(neighbourhoods) >= max_neighbourhoods:
            break

        if atom["type"] != "optimal_coverage":
            continue

        day = atom.get("day")
        shift = atom.get("shift")
        skill = atom.get("skill")

        if not day or not shift or not skill:
            continue

        off_qualified = get_off_qualified_nurses(instance, roster, day, skill)

        involved_nurses = [
            nurse for nurse in instance["sets"]["N"]
            if roster[nurse][day] is not None
        ]

        selected_nurses = list(dict.fromkeys(involved_nurses + off_qualified))
        selected_nurses = selected_nurses[:MAX_NURSES_PER_NEIGHBOURHOOD]

        days = expand_day_window(instance, day, radius=2)

        key = ("optimal_coverage", tuple(selected_nurses), tuple(days), shift, skill)

        if key in seen:
            continue

        seen.add(key)

        neighbourhoods.append({
            "neighbourhood_id": len(neighbourhoods) + 1,
            "target_penalty": "optimal_coverage",
            "target_cost": atom.get("cost", 0),
            "target_nurses": [],
            "nurses": selected_nurses,
            "days": days,
            "target_shift": shift,
            "target_skill": skill,
            "allowed_move_types": [
                "block_reassign",
                "block_swap",
                "reassign",
                "swap",
                "coverage_fill",
            ],
            "reason": atom["text"],
        })

    return neighbourhoods


# ============================================================
# Move pre-check and application
# ============================================================

def quick_move_precheck(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    move: Dict[str, Any],
) -> bool:
    if not isinstance(move, dict):
        return False

    nurses = set(instance["sets"]["N"])
    days = set(instance["sets"]["D"])
    shifts = set(instance["sets"]["S"])
    skills = set(instance["sets"]["K"])
    qualifications = instance["parameters"]["qualifications"]

    move_type = move.get("type")

    try:
        if move_type == "reassign":
            day = move["day"]
            src = move["from_nurse"]
            dst = move["to_nurse"]
            shift = move["shift"]
            skill = move["skill"]

            return (
                day in days
                and src in nurses
                and dst in nurses
                and shift in shifts
                and skill in skills
                and roster[src][day] == (shift, skill)
                and roster[dst][day] is None
                and qualifications[dst].get(skill, 0) == 1
            )

        if move_type == "swap":
            day = move["day"]
            nurse_a = move["nurse_a"]
            nurse_b = move["nurse_b"]

            if day not in days or nurse_a not in nurses or nurse_b not in nurses:
                return False

            if nurse_a == nurse_b:
                return False

            assn_a = roster[nurse_a][day]
            assn_b = roster[nurse_b][day]

            if assn_a is None and assn_b is None:
                return False

            if assn_a is not None:
                _, skill_a = assn_a
                if qualifications[nurse_b].get(skill_a, 0) != 1:
                    return False

            if assn_b is not None:
                _, skill_b = assn_b
                if qualifications[nurse_a].get(skill_b, 0) != 1:
                    return False

            return True

        if move_type == "coverage_fill":
            day = move["day"]
            shift = move["shift"]
            skill = move["skill"]
            nurse = move["to_nurse"]

            return (
                day in days
                and nurse in nurses
                and shift in shifts
                and skill in skills
                and roster[nurse][day] is None
                and qualifications[nurse].get(skill, 0) == 1
            )

        if move_type == "block_reassign":
            src = move["from_nurse"]
            dst = move["to_nurse"]
            block_days = move["days"]

            if src not in nurses or dst not in nurses or src == dst:
                return False

            if not isinstance(block_days, list) or not block_days:
                return False

            moved_any = False

            for day in block_days:
                if day not in days:
                    return False

                assn = roster[src][day]

                if assn is None:
                    continue

                moved_any = True

                _, skill = assn

                if roster[dst][day] is not None:
                    return False

                if qualifications[dst].get(skill, 0) != 1:
                    return False

            return moved_any

        if move_type == "block_swap":
            nurse_a = move["nurse_a"]
            nurse_b = move["nurse_b"]
            block_days = move["days"]

            if nurse_a not in nurses or nurse_b not in nurses or nurse_a == nurse_b:
                return False

            if not isinstance(block_days, list) or not block_days:
                return False

            swapped_any = False

            for day in block_days:
                if day not in days:
                    return False

                assn_a = roster[nurse_a][day]
                assn_b = roster[nurse_b][day]

                if assn_a is not None or assn_b is not None:
                    swapped_any = True

                if assn_a is not None:
                    _, skill_a = assn_a
                    if qualifications[nurse_b].get(skill_a, 0) != 1:
                        return False

                if assn_b is not None:
                    _, skill_b = assn_b
                    if qualifications[nurse_a].get(skill_b, 0) != 1:
                        return False

            return swapped_any

        if move_type == "remove_block":
            nurse = move["nurse"]
            block_days = move["days"]

            if nurse not in nurses:
                return False

            if not isinstance(block_days, list) or not block_days:
                return False

            removed_any = False

            for day in block_days:
                if day not in days:
                    return False

                if roster[nurse][day] is not None:
                    removed_any = True

            return removed_any

    except Exception:
        return False

    return False


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
            day = move["day"]
            src = move["from_nurse"]
            dst = move["to_nurse"]

            trial[dst][day] = trial[src][day]
            trial[src][day] = None
            return trial

        if move_type == "swap":
            day = move["day"]
            nurse_a = move["nurse_a"]
            nurse_b = move["nurse_b"]

            trial[nurse_a][day], trial[nurse_b][day] = (
                trial[nurse_b][day],
                trial[nurse_a][day],
            )
            return trial

        if move_type == "coverage_fill":
            day = move["day"]
            shift = move["shift"]
            skill = move["skill"]
            nurse = move["to_nurse"]

            trial[nurse][day] = (shift, skill)
            return trial

        if move_type == "block_reassign":
            src = move["from_nurse"]
            dst = move["to_nurse"]

            for day in move["days"]:
                if trial[src][day] is None:
                    continue

                trial[dst][day] = trial[src][day]
                trial[src][day] = None

            return trial

        if move_type == "block_swap":
            nurse_a = move["nurse_a"]
            nurse_b = move["nurse_b"]

            for day in move["days"]:
                trial[nurse_a][day], trial[nurse_b][day] = (
                    trial[nurse_b][day],
                    trial[nurse_a][day],
                )

            return trial

        if move_type == "remove_block":
            nurse = move["nurse"]

            for day in move["days"]:
                trial[nurse][day] = None

            return trial

    except Exception:
        return None

    return None


def apply_bundle(
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    bundle: Dict[str, Any],
    instance: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]]:
    if not isinstance(bundle, dict):
        return None

    moves = bundle.get("moves")

    if not isinstance(moves, list) or len(moves) == 0:
        return None

    trial = deep_copy_roster(roster)

    for move in moves[:MAX_MOVES_PER_BUNDLE]:
        if not quick_move_precheck(instance, trial, move):
            return None

        next_trial = apply_move(trial, move)

        if next_trial is None:
            return None

        trial = next_trial

    return trial


# ============================================================
# LLM / agent
# ============================================================

def extract_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            obj = json.loads(snippet)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    return []


def call_llm(client, model: str, prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
        max_output_tokens=10000,
        stream=False,
    )
    return response.output_text or ""


def build_agentic_repair_prompt(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    neighbourhoods: List[Dict[str, Any]],
    rejected_memory: Optional[List[Dict[str, Any]]] = None,
) -> str:
    atoms = extract_penalty_atoms(instance, validator_text)
    workload_stats = compute_nurse_workload_stats(instance, roster)

    compact_roster = {
        nurse: {
            day: (
                "OFF"
                if roster[nurse][day] is None
                else f"{roster[nurse][day][0]}/{roster[nurse][day][1]}"
            )
            for day in instance["sets"]["D"]
        }
        for nurse in instance["sets"]["N"]
    }

    payload = {
        "current_cost_summary": parse_validator_output(validator_text),
        "hard_constraint_context": build_hard_constraint_context(instance),
        "top_penalty_atoms": atoms[:30],
        "workload_stats": workload_stats,
        "candidate_neighbourhoods": neighbourhoods,
        "current_roster_compact": compact_roster,
        "qualifications": instance["parameters"]["qualifications"],
        "contracts": instance["parameters"]["contracts"],
        "coverage": instance["parameters"]["coverage"],
        "shift_successions": instance["parameters"].get("shift_successions", {}),
        "historical_assignments": instance["parameters"].get("historical_assignments", {}),
        "previous_rejected_attempts": rejected_memory or [],
    }

    template = load_prompt_template(AGENT_PROMPT_TEMPLATE_PATH)

    prompt = template.replace("{{MAX_AGENT_BUNDLES}}", str(MAX_AGENT_BUNDLES))
    prompt = prompt.replace("{{MAX_MOVES_PER_BUNDLE}}", str(MAX_MOVES_PER_BUNDLE))
    prompt = prompt.replace("{{PAYLOAD_JSON}}", json.dumps(payload, indent=2))

    return prompt


def save_llm_call(
    iteration: int,
    summary: Dict[str, Any],
    built_prompt: str,
    raw_response: str,
    parsed_bundles: List[Dict[str, Any]],
) -> None:
    base = f"iter_{iteration:04d}"

    with open(os.path.join(LLM_CALLS_DIR, f"{base}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(LLM_CALLS_DIR, f"{base}_built_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(built_prompt)

    with open(os.path.join(LLM_CALLS_DIR, f"{base}_raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw_response)

    with open(os.path.join(LLM_CALLS_DIR, f"{base}_parsed.json"), "w", encoding="utf-8") as f:
        json.dump(parsed_bundles, f, indent=2)


def ask_agent_for_repair_bundles(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    iteration: int,
    rejected_memory: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not API_KEY:
        raise RuntimeError("Missing NVIDIA_API_KEY environment variable")

    client = OpenAI(api_key=API_KEY)
    neighbourhoods = generate_block_repair_neighbourhoods(
        instance=instance,
        roster=roster,
        validator_text=validator_text,
        max_neighbourhoods=MAX_NEIGHBOURHOODS,
    )

    prompt = build_agentic_repair_prompt(
        instance=instance,
        roster=roster,
        validator_text=validator_text,
        neighbourhoods=neighbourhoods,
        rejected_memory=rejected_memory,
    )

    prompt_record = {
        "model": MODEL,
        "iteration": iteration,
        "neighbourhood_count": len(neighbourhoods),
        "neighbourhoods": neighbourhoods,
        "rejected_memory_count": len(rejected_memory or []),
    }

    try:
        content = call_llm(client, MODEL, prompt)
        bundles = extract_json_array(content)

        cleaned = []

        for i, bundle in enumerate(bundles[:MAX_AGENT_BUNDLES], start=1):
            if not isinstance(bundle, dict):
                continue

            moves = bundle.get("moves")

            if not isinstance(moves, list) or not moves:
                continue

            bundle["bundle_id"] = bundle.get("bundle_id", i)
            bundle["moves"] = moves[:MAX_MOVES_PER_BUNDLE]

            cleaned.append(bundle)

        save_llm_call(iteration, prompt_record, prompt, content, cleaned)
        return cleaned

    except Exception as e:
        save_llm_call(iteration, prompt_record, prompt, f"ERROR: {e}", [])
        return []


# ============================================================
# Logging and candidate evaluation
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
    base = f"call_{iteration:04d}_bundle_{bundle_index:02d}"

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}_bundle.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)


def save_search_log(log_data: Dict[str, Any]) -> None:
    with open(os.path.join(OUTPUT_DIR, "search_log.json"), "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)


def save_rejected_memory(rejected_memory: List[Dict[str, Any]]) -> None:
    with open(os.path.join(OUTPUT_DIR, "rejected_memory.json"), "w", encoding="utf-8") as f:
        json.dump(rejected_memory, f, indent=2)


def save_validator_calls_index(rows: List[Dict[str, Any]]) -> None:
    path = os.path.join(OUTPUT_DIR, "validator_calls_index.csv")

    fieldnames = [
        "call_id",
        "iteration",
        "bundle_index",
        "bundle_id",
        "neighbourhood_id",
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
    rejected_memory: List[Dict[str, Any]],
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
        trial_roster = apply_bundle(current_roster, bundle, instance=instance)

        if trial_roster is None:
            rejected_memory.append({
                "iteration": iteration,
                "bundle_id": bundle.get("bundle_id", bundle_index),
                "neighbourhood_id": bundle.get("neighbourhood_id", ""),
                "reason": "failed_quick_precheck_or_apply",
                "bundle": bundle,
            })
            continue

        candidate_dir = os.path.join(
            TMP_DIR,
            f"iter_{iteration:04d}_bundle_{bundle_index:02d}",
        )

        try:
            raw, parsed = evaluate_roster_with_validator(instance, trial_roster, candidate_dir)
        except Exception as e:
            rejected_memory.append({
                "iteration": iteration,
                "bundle_id": bundle.get("bundle_id", bundle_index),
                "neighbourhood_id": bundle.get("neighbourhood_id", ""),
                "reason": "validator_error",
                "error": str(e),
                "bundle": bundle,
            })
            continue

        delta = parsed["total_cost"] - old_cost

        save_validator_call(iteration, bundle_index, raw, parsed, bundle)

        validator_index_rows.append({
            "call_id": f"{iteration:04d}_{bundle_index:02d}",
            "iteration": iteration,
            "bundle_index": bundle_index,
            "bundle_id": bundle.get("bundle_id", bundle_index),
            "neighbourhood_id": bundle.get("neighbourhood_id", ""),
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
            rejected_memory.append({
                "iteration": iteration,
                "bundle_id": bundle.get("bundle_id", bundle_index),
                "neighbourhood_id": bundle.get("neighbourhood_id", ""),
                "reason": "hard_constraint_violation",
                "hard_violations": {
                    "minimal_coverage": parsed["minimal_coverage"],
                    "required_skill": parsed["required_skill"],
                    "illegal_succession": parsed["illegal_succession"],
                    "single_assignment": parsed["single_assignment"],
                },
                "old_total_cost": old_cost,
                "new_total_cost": parsed["total_cost"],
                "delta": delta,
                "bundle": bundle,
            })
            continue

        if parsed["total_cost"] < old_cost:
            if best_result is None or bundle_sort_key(parsed) < bundle_sort_key(best_result):
                best_bundle = bundle
                best_roster = trial_roster
                best_raw = raw
                best_result = parsed
        else:
            rejected_memory.append({
                "iteration": iteration,
                "bundle_id": bundle.get("bundle_id", bundle_index),
                "neighbourhood_id": bundle.get("neighbourhood_id", ""),
                "reason": "not_cost_improving",
                "old_total_cost": old_cost,
                "new_total_cost": parsed["total_cost"],
                "delta": delta,
                "cost_breakdown": {
                    "total_assignment_cost": parsed["total_assignment_cost"],
                    "consecutive_cost": parsed["consecutive_cost"],
                    "non_working_days_cost": parsed["non_working_days_cost"],
                    "preferences_cost": parsed["preferences_cost"],
                    "max_working_weekend_cost": parsed["max_working_weekend_cost"],
                    "complete_weekends_cost": parsed["complete_weekends_cost"],
                    "optimal_coverage_cost": parsed["optimal_coverage_cost"],
                },
                "bundle": bundle,
            })

    return best_bundle, best_roster, best_raw, best_result


# ============================================================
# Main pipeline
# ============================================================

def run_agentic_block_improvement() -> None:
    ensure_dirs()

    if not os.path.exists(INSTANCE_PATH):
        print(f"Missing instance file: {INSTANCE_PATH}")
        return

    if not os.path.exists(AGENT_PROMPT_TEMPLATE_PATH):
        print(f"Missing prompt file: {AGENT_PROMPT_TEMPLATE_PATH}")
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
    rejected_memory: List[Dict[str, Any]] = []

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
        print(f"\nIteration {iteration}")

        penalty_atoms = extract_penalty_atoms(instance, current_raw)
        neighbourhoods = generate_block_repair_neighbourhoods(
            instance=instance,
            roster=current_roster,
            validator_text=current_raw,
        )

        print(f"Extracted penalty atoms: {len(penalty_atoms)}")
        print(f"Generated neighbourhoods: {len(neighbourhoods)}")
        print(f"Rejected memory available: {len(rejected_memory[-MAX_REJECTED_MEMORY:])}")

        bundles = ask_agent_for_repair_bundles(
            instance=instance,
            roster=current_roster,
            validator_text=current_raw,
            iteration=iteration,
            rejected_memory=rejected_memory[-MAX_REJECTED_MEMORY:],
        )

        print(f"Agent returned bundles: {len(bundles)}")

        if not bundles:
            search_log["iterations"].append({
                "iteration": iteration,
                "status": "no_agent_bundles",
                "old_cost": current_result["total_cost"],
                "new_cost": current_result["total_cost"],
                "delta": 0,
                "changes": [],
            })
            break

        old_cost = current_result["total_cost"]

        best_bundle, next_roster, next_raw, next_result = choose_best_bundle(
            instance=instance,
            current_roster=current_roster,
            current_result=current_result,
            bundles=bundles,
            iteration=iteration,
            validator_index_rows=validator_index_rows,
            rejected_memory=rejected_memory,
        )

        search_log["total_validator_calls"] = len(validator_index_rows)

        if best_bundle is None or next_roster is None or next_raw is None or next_result is None:
            print(f"Iteration {iteration}: no improving hard-feasible bundle found.")

            search_log["iterations"].append({
                "iteration": iteration,
                "status": "no_improving_bundle_found",
                "accepted_bundle_id": None,
                "old_cost": old_cost,
                "new_cost": old_cost,
                "delta": 0,
                "changes": [],
                "rejected_memory_size": len(rejected_memory),
            })

            save_search_log(search_log)
            save_rejected_memory(rejected_memory)
            continue

        changes = diff_rosters(current_roster, next_roster)
        accepted_changes.extend(changes)

        current_roster = next_roster
        current_result = next_result
        current_raw = next_raw

        new_cost = next_result["total_cost"]
        delta = new_cost - old_cost

        print(
            f"Iteration {iteration}: accepted bundle {best_bundle.get('bundle_id')} "
            f"cost {old_cost} -> {new_cost}, delta {delta}"
        )

        for change in changes:
            print("  " + change)

        search_log["iterations"].append({
            "iteration": iteration,
            "status": "improved",
            "accepted_bundle_id": best_bundle.get("bundle_id", iteration),
            "accepted_neighbourhood_id": best_bundle.get("neighbourhood_id", ""),
            "accepted_bundle_reason": best_bundle.get("reason", ""),
            "old_cost": old_cost,
            "new_cost": new_cost,
            "delta": delta,
            "changes": changes,
            "accepted_bundle": best_bundle,
            "rejected_memory_size": len(rejected_memory),
        })

        if current_result["total_cost"] < best_result["total_cost"]:
            best_roster = deep_copy_roster(current_roster)
            best_raw = current_raw
            best_result = current_result

            search_log["best_total_cost"] = best_result["total_cost"]
            save_best_outputs(instance, initial_raw, best_raw, best_roster)

        save_search_log(search_log)
        save_rejected_memory(rejected_memory)

    save_search_log(search_log)
    save_rejected_memory(rejected_memory)
    save_validator_calls_index(validator_index_rows)
    save_best_outputs(instance, initial_raw, best_raw, best_roster)

    print("\n==============================")
    print("Search finished")
    print("==============================")
    print(f"Initial cost: {initial_result['total_cost']}")
    print(f"Best cost:    {best_result['total_cost']}")
    print(f"Improvement:  {initial_result['total_cost'] - best_result['total_cost']}")
    print(f"Validator calls: {len(validator_index_rows)}")
    print(f"Rejected attempts stored: {len(rejected_memory)}")

    if accepted_changes:
        print("\nAccepted changes:")
        for line in accepted_changes:
            print(line)
    else:
        print("\nNo changes accepted.")

    print(f"\nOutputs saved in: {OUTPUT_DIR}")


def main() -> None:
    random.seed(RANDOM_SEED)
    run_agentic_block_improvement()


if __name__ == "__main__":
    main()