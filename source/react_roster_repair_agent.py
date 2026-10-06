
import os
import re
import csv
import json
import copy
import glob
import shutil
import subprocess
from typing import Dict, List, Tuple, Optional, Any

from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

MAX_REACT_STEPS = 10
MAX_MOVES_PER_BUNDLE = 10
MAX_OBSERVATION_HISTORY = 8
MAX_TOP_ATOMS = 35

PROMPT_TEMPLATE_PATH = "react_repair_prompt.txt"

INSTANCE_PATH = "response1.json"
INITIAL_ROSTER_DIR = "initial_roster"
INITIAL_VALIDATOR_OUTPUT_PATH = os.path.join(INITIAL_ROSTER_DIR, "validator_output.txt")

OUTPUT_DIR = "improved_roster_react_agent"
TMP_DIR = "tmp_react_candidates"
REACT_TRACE_DIR = os.path.join(OUTPUT_DIR, "react_trace")
VALIDATOR_CALLS_DIR = os.path.join(OUTPUT_DIR, "validator_calls")

VALIDATOR_DIR = "validator"
VALIDATOR_WEEKS_DIR = os.path.join(VALIDATOR_DIR, "weeks")
VALIDATOR_JAR = os.path.join(VALIDATOR_DIR, "validator.jar")
VALIDATOR_SCENARIO_FILE = os.path.join(VALIDATOR_DIR, "scenario.json")
VALIDATOR_HISTORY_FILE = os.path.join(VALIDATOR_DIR, "history.json")

API_KEY = ""
BASE_URL = "https://integrate.api.nvidia.com/v1"
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


# ============================================================
# Basic helpers
# ============================================================

def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(REACT_TRACE_DIR, exist_ok=True)
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
# Load instance and initial roster
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

        for assignment in payload.get("assignments", []):
            nurse = assignment["nurse"]
            day_short = assignment["day"]
            shift_full = assignment["shiftType"]
            skill = assignment["skill"]

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
# Export and validator
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
# Penalty atoms and observation
# ============================================================

def day_index_to_label(instance: Dict[str, Any], day_index: Optional[int]) -> Optional[str]:
    if day_index is None:
        return None

    days = instance["sets"]["D"]
    if 0 <= day_index < len(days):
        return days[day_index]

    return None


def extract_penalty_atoms(instance: Dict[str, Any], validator_text: str) -> List[Dict[str, Any]]:
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
            "day": day_label,
            "shift": shift,
            "skill": skill,
            "missing": missing,
            "cost": cost,
            "text": line,
        })

    atoms.sort(key=lambda x: x.get("cost", 0), reverse=True)
    return atoms


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


def compact_roster_for_prompt(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    atoms: List[Dict[str, Any]],
) -> Dict[str, Dict[str, str]]:
    focus_nurses = []
    focus_days = []

    for atom in atoms[:MAX_TOP_ATOMS]:
        nurse = atom.get("nurse")
        day = atom.get("day")

        if nurse and nurse not in focus_nurses:
            focus_nurses.append(nurse)

        if day and day not in focus_days:
            focus_days.append(day)

    # Add some complete context but keep prompt size bounded.
    focus_nurses = focus_nurses[:12]
    focus_days = sorted(focus_days[:18], key=day_sort_key)

    compact = {}

    for nurse in focus_nurses:
        compact[nurse] = {}
        for day in focus_days:
            assn = roster[nurse][day]
            compact[nurse][day] = "OFF" if assn is None else f"{assn[0]}/{assn[1]}"

    return compact


def cost_delta(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, int]:
    keys = [
        "total_cost",
        "total_assignment_cost",
        "consecutive_cost",
        "non_working_days_cost",
        "preferences_cost",
        "max_working_weekend_cost",
        "complete_weekends_cost",
        "optimal_coverage_cost",
        "minimal_coverage",
        "required_skill",
        "illegal_succession",
        "single_assignment",
    ]

    return {k: int(new.get(k, 0)) - int(old.get(k, 0)) for k in keys}


def build_observation(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    validator_text: str,
    parsed: Dict[str, Any],
    step: int,
    best_cost: int,
    previous_observations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    atoms = extract_penalty_atoms(instance, validator_text)
    workload = compute_nurse_workload_stats(instance, roster)

    return {
        "step": step,
        "current_cost_summary": parsed,
        "best_cost_so_far": best_cost,
        "top_penalty_atoms": atoms[:MAX_TOP_ATOMS],
        "workload_stats": workload,
        "current_roster_focus": compact_roster_for_prompt(instance, roster, atoms),
        "valid_sets": {
            "nurses": instance["sets"]["N"],
            "days": instance["sets"]["D"],
            "shifts": instance["sets"]["S"],
            "skills": instance["sets"]["K"],
        },
        "qualifications": instance["parameters"]["qualifications"],
        "contracts": instance["parameters"]["contracts"],
        "recent_observations": previous_observations[-MAX_OBSERVATION_HISTORY:],
    }


# ============================================================
# Move validation and application
# ============================================================

def is_qualified(instance: Dict[str, Any], nurse: str, skill: str) -> bool:
    return instance["parameters"]["qualifications"][nurse].get(skill, 0) == 1


def quick_move_precheck(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    move: Dict[str, Any],
) -> Tuple[bool, str]:
    nurses = set(instance["sets"]["N"])
    days = set(instance["sets"]["D"])
    shifts = set(instance["sets"]["S"])
    skills = set(instance["sets"]["K"])

    if not isinstance(move, dict):
        return False, "move_not_dict"

    try:
        t = move.get("type")

        if t == "reassign":
            day = move["day"]
            src = move["from_nurse"]
            dst = move["to_nurse"]
            shift = move["shift"]
            skill = move["skill"]

            if day not in days:
                return False, f"invalid_day:{day}"
            if src not in nurses or dst not in nurses:
                return False, "invalid_nurse"
            if src == dst:
                return False, "same_source_destination"
            if shift not in shifts or skill not in skills:
                return False, "invalid_shift_or_skill"
            if roster[src][day] != (shift, skill):
                return False, "source_assignment_does_not_match"
            if roster[dst][day] is not None:
                return False, "destination_already_working"
            if not is_qualified(instance, dst, skill):
                return False, "destination_not_qualified"
            return True, "ok"

        if t == "swap":
            day = move["day"]
            a = move["nurse_a"]
            b = move["nurse_b"]

            if day not in days:
                return False, f"invalid_day:{day}"
            if a not in nurses or b not in nurses:
                return False, "invalid_nurse"
            if a == b:
                return False, "same_nurse_swap"

            assn_a = roster[a][day]
            assn_b = roster[b][day]

            if assn_a is None and assn_b is None:
                return False, "both_off"

            if assn_a is not None and not is_qualified(instance, b, assn_a[1]):
                return False, "nurse_b_not_qualified_for_a_assignment"

            if assn_b is not None and not is_qualified(instance, a, assn_b[1]):
                return False, "nurse_a_not_qualified_for_b_assignment"

            return True, "ok"

        if t == "coverage_fill":
            day = move["day"]
            shift = move["shift"]
            skill = move["skill"]
            nurse = move["to_nurse"]

            if day not in days:
                return False, f"invalid_day:{day}"
            if nurse not in nurses:
                return False, "invalid_nurse"
            if shift not in shifts or skill not in skills:
                return False, "invalid_shift_or_skill"
            if roster[nurse][day] is not None:
                return False, "nurse_already_working"
            if not is_qualified(instance, nurse, skill):
                return False, "nurse_not_qualified"
            return True, "ok"

        if t == "block_reassign":
            src = move["from_nurse"]
            dst = move["to_nurse"]
            block_days = move["days"]

            if src not in nurses or dst not in nurses:
                return False, "invalid_nurse"
            if src == dst:
                return False, "same_source_destination"
            if not isinstance(block_days, list) or not block_days:
                return False, "invalid_days_list"

            moved_any = False

            for day in block_days:
                if day not in days:
                    return False, f"invalid_day:{day}"

                assn = roster[src][day]
                if assn is None:
                    continue

                moved_any = True
                _, skill = assn

                if roster[dst][day] is not None:
                    return False, f"destination_already_working:{dst}:{day}"

                if not is_qualified(instance, dst, skill):
                    return False, f"destination_not_qualified:{dst}:{skill}"

            if not moved_any:
                return False, "block_reassign_moves_nothing"

            return True, "ok"

        if t == "block_swap":
            a = move["nurse_a"]
            b = move["nurse_b"]
            block_days = move["days"]

            if a not in nurses or b not in nurses:
                return False, "invalid_nurse"
            if a == b:
                return False, "same_nurse_swap"
            if not isinstance(block_days, list) or not block_days:
                return False, "invalid_days_list"

            changed_any = False

            for day in block_days:
                if day not in days:
                    return False, f"invalid_day:{day}"

                assn_a = roster[a][day]
                assn_b = roster[b][day]

                if assn_a is not None or assn_b is not None:
                    changed_any = True

                if assn_a is not None and not is_qualified(instance, b, assn_a[1]):
                    return False, f"nurse_b_not_qualified:{b}:{assn_a[1]}"

                if assn_b is not None and not is_qualified(instance, a, assn_b[1]):
                    return False, f"nurse_a_not_qualified:{a}:{assn_b[1]}"

            if not changed_any:
                return False, "block_swap_changes_nothing"

            return True, "ok"

    except Exception as e:
        return False, f"exception:{e}"

    return False, "unknown_move_type"


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

        if t == "block_reassign":
            src = move["from_nurse"]
            dst = move["to_nurse"]

            for day in move["days"]:
                if trial[src][day] is None:
                    continue
                trial[dst][day] = trial[src][day]
                trial[src][day] = None

            return trial

        if t == "block_swap":
            a = move["nurse_a"]
            b = move["nurse_b"]

            for day in move["days"]:
                trial[a][day], trial[b][day] = trial[b][day], trial[a][day]

            return trial

    except Exception:
        return None

    return None


def apply_bundle(
    instance: Dict[str, Any],
    roster: Dict[str, Dict[str, Optional[Tuple[str, str]]]],
    bundle: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Dict[str, Optional[Tuple[str, str]]]]], List[Dict[str, str]]]:
    moves = bundle.get("moves", [])

    if not isinstance(moves, list) or not moves:
        return None, [{"status": "failed", "reason": "empty_or_invalid_moves"}]

    if len(moves) > MAX_MOVES_PER_BUNDLE:
        moves = moves[:MAX_MOVES_PER_BUNDLE]

    trial = deep_copy_roster(roster)
    precheck_log = []

    for i, move in enumerate(moves, start=1):
        ok, reason = quick_move_precheck(instance, trial, move)
        precheck_log.append({
            "move_index": str(i),
            "move_type": str(move.get("type")),
            "status": "ok" if ok else "failed",
            "reason": reason,
        })

        if not ok:
            return None, precheck_log

        next_trial = apply_move(trial, move)

        if next_trial is None:
            precheck_log[-1]["status"] = "failed"
            precheck_log[-1]["reason"] = "apply_move_returned_none"
            return None, precheck_log

        trial = next_trial

    return trial, precheck_log


# ============================================================
# ReAct agent
# ============================================================

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


def build_react_prompt(observation: Dict[str, Any]) -> str:
    template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    return template.replace("{{OBSERVATION_JSON}}", json.dumps(observation, indent=2))


def call_react_agent(observation: Dict[str, Any], step: int) -> Dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("Missing NVIDIA_API_KEY environment variable")

    client = OpenAI(api_key=API_KEY)
    prompt = build_react_prompt(observation)

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        temperature=0.2,
        max_output_tokens=4000,
        stream=False,
    )

    raw = response.output_text or ""
    parsed = extract_json_object(raw)

    save_react_call(step, prompt, raw, parsed)

    if not parsed:
        return {
            "thought": "The response was not valid JSON.",
            "action": {
                "type": "invalid",
                "reason": "invalid_json",
            },
        }

    return parsed


def save_react_call(step: int, prompt: str, raw: str, parsed: Optional[Dict[str, Any]]) -> None:
    base = f"step_{step:04d}"

    with open(os.path.join(REACT_TRACE_DIR, f"{base}_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    with open(os.path.join(REACT_TRACE_DIR, f"{base}_raw.txt"), "w", encoding="utf-8") as f:
        f.write(raw)

    with open(os.path.join(REACT_TRACE_DIR, f"{base}_parsed.json"), "w", encoding="utf-8") as f:
        json.dump(parsed or {}, f, indent=2)


# ============================================================
# Logging
# ============================================================

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


def save_validator_call(step: int, raw_text: str, parsed: Dict[str, Any], action: Dict[str, Any]) -> None:
    base = f"step_{step:04d}"

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}_parsed.json"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)

    with open(os.path.join(VALIDATOR_CALLS_DIR, f"{base}_action.json"), "w", encoding="utf-8") as f:
        json.dump(action, f, indent=2)


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


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_trace_csv(path: str, row: Dict[str, Any]) -> None:
    fieldnames = [
        "step",
        "action_type",
        "bundle_id",
        "accepted",
        "status",
        "old_cost",
        "new_cost",
        "delta",
        "hard_ok",
        "minimal_coverage",
        "required_skill",
        "illegal_succession",
        "single_assignment",
        "total_assignment_cost_delta",
        "consecutive_cost_delta",
        "non_working_days_cost_delta",
        "max_working_weekend_cost_delta",
        "optimal_coverage_cost_delta",
        "move_count",
        "target_penalty",
        "reason",
    ]

    exists = os.path.exists(path)

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Main ReAct loop
# ============================================================

def run_react_roster_repair() -> None:
    ensure_dirs()

    if not os.path.exists(INSTANCE_PATH):
        print(f"Missing instance file: {INSTANCE_PATH}")
        return

    if not os.path.exists(PROMPT_TEMPLATE_PATH):
        print(f"Missing prompt template: {PROMPT_TEMPLATE_PATH}")
        return

    if not API_KEY:
        print("Missing NVIDIA_API_KEY environment variable.")
        return

    try:
        instance = load_instance(INSTANCE_PATH)
        partial_roster = load_initial_roster(INITIAL_ROSTER_DIR)
        current_roster = normalize_roster(instance, partial_roster)
    except Exception as e:
        print(f"Failed to load inputs: {e}")
        return

    try:
        if os.path.exists(INITIAL_VALIDATOR_OUTPUT_PATH):
            current_raw = load_text_if_exists(INITIAL_VALIDATOR_OUTPUT_PATH)
            current_result = parse_validator_output(current_raw)
        else:
            current_raw, current_result = evaluate_roster_with_validator(
                instance,
                current_roster,
                os.path.join(TMP_DIR, "initial_eval"),
            )
    except Exception as e:
        print(f"Initial validation failed: {e}")
        return

    if not current_result["hard_ok"]:
        print("Initial roster is not hard-feasible. ReAct repair expects a hard-feasible starting roster.")
        print(current_result)
        return

    initial_cost = current_result["total_cost"]
    best_cost = current_result["total_cost"]
    best_roster = deep_copy_roster(current_roster)
    best_raw = current_raw
    best_result = current_result

    previous_observations: List[Dict[str, Any]] = []
    accepted_steps: List[Dict[str, Any]] = []

    save_best_outputs(instance, current_raw, best_raw, best_roster)

    print("Initial hard feasible:", current_result["hard_ok"])
    print("Initial total cost:", initial_cost)

    trace_csv_path = os.path.join(OUTPUT_DIR, "react_trace_index.csv")

    for step in range(1, MAX_REACT_STEPS + 1):
        print(f"\nReAct step {step}")
        print(f"Current cost: {current_result['total_cost']}")

        observation = build_observation(
            instance=instance,
            roster=current_roster,
            validator_text=current_raw,
            parsed=current_result,
            step=step,
            best_cost=best_cost,
            previous_observations=previous_observations,
        )

        save_json(os.path.join(REACT_TRACE_DIR, f"step_{step:04d}_observation_input.json"), observation)

        agent_output = call_react_agent(observation, step)
        action = agent_output.get("action", {})

        if action.get("type") == "stop":
            print("Agent stopped:", action.get("reason", ""))
            previous_observations.append({
                "step": step,
                "status": "agent_stopped",
                "reason": action.get("reason", ""),
            })
            break

        if action.get("type") != "propose_bundle":
            print("Invalid agent action.")
            previous_observations.append({
                "step": step,
                "status": "invalid_action",
                "agent_output": agent_output,
            })
            continue

        moves = action.get("moves", [])
        print(f"Agent proposed bundle with {len(moves)} moves.")
        print("Target penalty:", action.get("target_penalty", ""))

        old_result = current_result
        old_cost = old_result["total_cost"]

        trial_roster, precheck_log = apply_bundle(instance, current_roster, action)

        if trial_roster is None:
            observation_result = {
                "step": step,
                "status": "precheck_failed",
                "accepted": False,
                "reason": "Python quick precheck rejected the bundle before validator.",
                "precheck_log": precheck_log,
                "old_cost": old_cost,
                "new_cost": old_cost,
                "delta": 0,
                "action": action,
            }

            print("Precheck failed.")
            previous_observations.append(observation_result)
            save_json(os.path.join(REACT_TRACE_DIR, f"step_{step:04d}_observation_result.json"), observation_result)

            append_trace_csv(trace_csv_path, {
                "step": step,
                "action_type": action.get("type"),
                "bundle_id": action.get("bundle_id", ""),
                "accepted": False,
                "status": "precheck_failed",
                "old_cost": old_cost,
                "new_cost": old_cost,
                "delta": 0,
                "hard_ok": "",
                "minimal_coverage": "",
                "required_skill": "",
                "illegal_succession": "",
                "single_assignment": "",
                "total_assignment_cost_delta": "",
                "consecutive_cost_delta": "",
                "non_working_days_cost_delta": "",
                "max_working_weekend_cost_delta": "",
                "optimal_coverage_cost_delta": "",
                "move_count": len(moves),
                "target_penalty": action.get("target_penalty", ""),
                "reason": action.get("reason", ""),
            })
            continue

        try:
            candidate_raw, candidate_result = evaluate_roster_with_validator(
                instance,
                trial_roster,
                os.path.join(TMP_DIR, f"step_{step:04d}"),
            )
        except Exception as e:
            observation_result = {
                "step": step,
                "status": "validator_error",
                "accepted": False,
                "reason": str(e),
                "old_cost": old_cost,
                "new_cost": old_cost,
                "delta": 0,
                "action": action,
            }
            print("Validator error.")
            previous_observations.append(observation_result)
            save_json(os.path.join(REACT_TRACE_DIR, f"step_{step:04d}_observation_result.json"), observation_result)
            continue

        save_validator_call(step, candidate_raw, candidate_result, action)

        delta = candidate_result["total_cost"] - old_cost
        deltas = cost_delta(old_result, candidate_result)
        changes = diff_rosters(current_roster, trial_roster)

        accepted = candidate_result["hard_ok"] and candidate_result["total_cost"] < old_cost

        observation_result = {
            "step": step,
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "old_cost": old_cost,
            "new_cost": candidate_result["total_cost"],
            "delta": delta,
            "hard_ok": candidate_result["hard_ok"],
            "hard_violations": {
                "minimal_coverage": candidate_result["minimal_coverage"],
                "required_skill": candidate_result["required_skill"],
                "illegal_succession": candidate_result["illegal_succession"],
                "single_assignment": candidate_result["single_assignment"],
            },
            "cost_delta_by_category": deltas,
            "precheck_log": precheck_log,
            "move_count": len(moves),
            "changes": changes,
            "action": action,
            "agent_thought_summary": agent_output.get("thought", ""),
        }

        if accepted:
            print(f"Accepted: {old_cost} -> {candidate_result['total_cost']} delta {delta}")

            current_roster = trial_roster
            current_raw = candidate_raw
            current_result = candidate_result

            accepted_steps.append(observation_result)

            if candidate_result["total_cost"] < best_cost:
                best_cost = candidate_result["total_cost"]
                best_roster = deep_copy_roster(current_roster)
                best_raw = current_raw
                best_result = current_result
                save_best_outputs(instance, load_text_if_exists(INITIAL_VALIDATOR_OUTPUT_PATH) or current_raw, best_raw, best_roster)
        else:
            print(f"Rejected: hard_ok={candidate_result['hard_ok']} cost {old_cost} -> {candidate_result['total_cost']} delta {delta}")

        previous_observations.append(observation_result)
        save_json(os.path.join(REACT_TRACE_DIR, f"step_{step:04d}_observation_result.json"), observation_result)

        append_trace_csv(trace_csv_path, {
            "step": step,
            "action_type": action.get("type"),
            "bundle_id": action.get("bundle_id", ""),
            "accepted": accepted,
            "status": observation_result["status"],
            "old_cost": old_cost,
            "new_cost": candidate_result["total_cost"],
            "delta": delta,
            "hard_ok": candidate_result["hard_ok"],
            "minimal_coverage": candidate_result["minimal_coverage"],
            "required_skill": candidate_result["required_skill"],
            "illegal_succession": candidate_result["illegal_succession"],
            "single_assignment": candidate_result["single_assignment"],
            "total_assignment_cost_delta": deltas["total_assignment_cost"],
            "consecutive_cost_delta": deltas["consecutive_cost"],
            "non_working_days_cost_delta": deltas["non_working_days_cost"],
            "max_working_weekend_cost_delta": deltas["max_working_weekend_cost"],
            "optimal_coverage_cost_delta": deltas["optimal_coverage_cost"],
            "move_count": len(moves),
            "target_penalty": action.get("target_penalty", ""),
            "reason": action.get("reason", ""),
        })

    save_json(os.path.join(OUTPUT_DIR, "accepted_steps.json"), accepted_steps)
    save_json(os.path.join(OUTPUT_DIR, "all_observations.json"), previous_observations)
    save_best_outputs(instance, load_text_if_exists(INITIAL_VALIDATOR_OUTPUT_PATH) or best_raw, best_raw, best_roster)

    summary = {
        "initial_cost": initial_cost,
        "best_cost": best_result["total_cost"],
        "improvement": initial_cost - best_result["total_cost"],
        "accepted_step_count": len(accepted_steps),
        "final_best_result": best_result,
    }

    save_json(os.path.join(OUTPUT_DIR, "summary.json"), summary)

    print("\n==============================")
    print("ReAct repair finished")
    print("==============================")
    print(f"Initial cost: {initial_cost}")
    print(f"Best cost:    {best_result['total_cost']}")
    print(f"Improvement:  {initial_cost - best_result['total_cost']}")
    print(f"Accepted steps: {len(accepted_steps)}")
    print(f"Outputs saved in: {OUTPUT_DIR}")


def main() -> None:
    run_react_roster_repair()


if __name__ == "__main__":
    main()
