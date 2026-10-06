import json
import os
import re
from collections import defaultdict
from ortools.sat.python import cp_model


def safe_get(dct, *keys, default=None):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_day_label(day_label):
    m = re.match(r"^W(\d+)_(Mon|Tue|Wed|Thu|Fri|Sat|Sun)$", day_label)
    if not m:
        return None, None
    week_idx = int(m.group(1)) - 1
    day_name = m.group(2)
    return week_idx, day_name


def main():
    input_path = "response1.json"
    if not os.path.exists(input_path):
        print("Solver failed")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sets = data.get("sets", {})
    params = data.get("parameters", {})

    nurses = list(sets.get("N", []))
    days = list(sets.get("D", []))
    shifts = list(sets.get("S", []))
    skills = list(sets.get("K", []))

    coverage = params.get("coverage", {})
    qualifications = params.get("qualifications", {})
    contracts = params.get("contracts", {})
    shift_type_limits = params.get("shift_type_limits", {})
    shift_successions = params.get("shift_successions", {})
    historical = params.get("historical_assignments", {})
    preferences = params.get("preferences", [])
    weights = params.get("soft_rule_weights", {})

    w_opt_cov = int(weights.get("optimal_coverage", 30))
    w_prefs = int(weights.get("preferences", 10))
    w_consec_shift = int(weights.get("consecutive_shift_assignments", 15))
    w_consec_work = int(weights.get("consecutive_working_days", 30))
    w_consec_off = int(weights.get("consecutive_days_off", 30))
    w_complete_weekend = int(weights.get("complete_weekend", 30))
    w_total_assign = int(weights.get("total_assignments", 20))
    w_total_weekends = int(weights.get("total_working_weekends", 30))

    scenario_name = (
        data.get("scenario")
        or safe_get(data, "metadata", "scenario")
        or safe_get(data, "parameters", "scenario")
        or "unknown_scenario"
    )

    # Validate required sets
    if not nurses or not days or not shifts or not skills:
        print("Solver failed")
        return

    # Determine weeks and day-of-week mapping
    day_to_week = {}
    day_to_dow = {}
    weeks_set = set()
    for d in days:
        wi, dow = parse_day_label(d)
        if wi is None:
            # If day labels are not in Wk_Day format, weekly export cannot be exact.
            # This instance uses W1_Mon etc.
            print("Solver failed")
            return
        day_to_week[d] = wi
        day_to_dow[d] = dow
        weeks_set.add(wi)
    num_weeks = max(weeks_set) + 1 if weeks_set else 0

    dow_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

    # Precompute weekend day indices per week
    week_to_sat = {}
    week_to_sun = {}
    for d in days:
        wi = day_to_week[d]
        if day_to_dow[d] == "Sat":
            week_to_sat[wi] = d
        elif day_to_dow[d] == "Sun":
            week_to_sun[wi] = d

    model = cp_model.CpModel()

    # Primary decision variables: x[n,d,s,k] in {0,1}
    x = {}
    for n in nurses:
        for d in days:
            for s in shifts:
                for k in skills:
                    x[(n, d, s, k)] = model.NewIntVar(0, 1, f"x[{n},{d},{s},{k}]")

    # Mandatory linking variables
    shift_work = {}
    for n in nurses:
        for d in days:
            for s in shifts:
                shift_work[(n, d, s)] = model.NewIntVar(0, 1, f"shift_work[{n},{d},{s}]")

    work = {}
    off = {}
    for n in nurses:
        for d in days:
            work[(n, d)] = model.NewIntVar(0, 1, f"work[{n},{d}]")
            off[(n, d)] = model.NewIntVar(0, 1, f"off[{n},{d}]")

    # Linking constraints:
    # 1) shift_work[n,d,s] = sum_k x[n,d,s,k]
    for n in nurses:
        for d in days:
            for s in shifts:
                model.Add(shift_work[(n, d, s)] == sum(x[(n, d, s, k)] for k in skills))

    # 2) work[n,d] = sum_s shift_work[n,d,s]
    for n in nurses:
        for d in days:
            model.Add(work[(n, d)] == sum(shift_work[(n, d, s)] for s in shifts))

    # 3) off[n,d] = 1 - work[n,d]
    for n in nurses:
        for d in days:
            model.Add(off[(n, d)] + work[(n, d)] == 1)

    # -------------------------
    # HARD CONSTRAINTS
    # -------------------------

    # H1. Single assignment per day: sum_{s,k} x[n,d,s,k] <= 1
    for n in nurses:
        for d in days:
            model.Add(sum(x[(n, d, s, k)] for s in shifts for k in skills) <= 1)

    # H2. Minimum coverage: sum_n x[n,d,s,k] >= min
    # Use all input coverage data; if some day/shift/skill missing, assume min=0 opt=0.
    for d in days:
        for s in shifts:
            for k in skills:
                min_req = int(safe_get(coverage, d, s, k, "min", default=0) or 0)
                model.Add(sum(x[(n, d, s, k)] for n in nurses) >= min_req)

    # H4. Skill compatibility: if qualifications[n][k] == 0 then x[n,d,s,k] == 0
    for n in nurses:
        qn = qualifications.get(n, {})
        for k in skills:
            if int(qn.get(k, 0) or 0) == 0:
                for d in days:
                    for s in shifts:
                        model.Add(x[(n, d, s, k)] == 0)

    # H3 Shift Successions (Hard Constraint)
    forbidden_pairs = shift_successions.get("forbidden", [])
    forbidden_pairs = [(a, b) for a, b in forbidden_pairs if a in shifts and b in shifts]

    for n in nurses:
        for di in range(len(days) - 1):
            d_prev = days[di]
            d_curr = days[di + 1]
            for a, b in forbidden_pairs:
                model.Add(shift_work[(n, d_prev, a)] + shift_work[(n, d_curr, b)] <= 1)

    for n in nurses:
        first_day = days[0]
        last_shift = safe_get(historical, n, "lastAssignedShiftType", default="None")
        if last_shift is None:
            last_shift = "None"
        if last_shift != "None":
            for a, b in forbidden_pairs:
                if a == last_shift:
                    model.Add(shift_work[(n, first_day, b)] == 0)

    # -------------------------
    # SOFT CONSTRAINTS (penalties)
    # -------------------------
    penalty_terms = []

    # S1. Optimal coverage (shortage only)
    for d in days:
        for s in shifts:
            for k in skills:
                opt_req = int(safe_get(coverage, d, s, k, "opt", default=0) or 0)
                if opt_req <= 0:
                    continue
                staffed = sum(x[(n, d, s, k)] for n in nurses)
                shortage = model.NewIntVar(0, len(nurses), f"short_opt[{d},{s},{k}]")
                model.Add(shortage >= opt_req - staffed)
                model.Add(shortage >= 0)
                penalty_terms.append(w_opt_cov * shortage)

    # Helper for consecutive penalties using automaton on binary sequence with historical prefix
    def add_consecutive_penalties_binary(seq_vars, hist_ones, hist_zeros, min_len, max_len, weight, name_prefix):
        # Penalize:
        # - runs of 1s shorter than min_len (missing days)
        # - runs of 1s longer than max_len (extra days)
        # similarly for 0s if seq_vars represent off; caller controls.
        # This function assumes seq_vars are 0/1 IntVars.
        if min_len is None:
            min_len = 0
        if max_len is None:
            max_len = len(seq_vars) + max(hist_ones, hist_zeros)
        min_len = int(min_len)
        max_len = int(max_len)

        # Extra above max: for each window of length max_len+1, if all ones then at least 1 extra.
        if max_len >= 0 and max_len + 1 <= len(seq_vars):
            for start in range(0, len(seq_vars) - (max_len + 1) + 1):
                window = seq_vars[start : start + max_len + 1]
                extra = model.NewIntVar(0, 1, f"{name_prefix}_extra[{start}]")
                model.Add(sum(window) - (max_len) <= extra)
                # If sum(window) == max_len+1 then extra >=1; else extra can be 0.
                penalty_terms.append(weight * extra)
        else:
            # If horizon shorter than max+1, no "extra" can be detected within horizon.
            pass

        # Missing below min: for each possible run start, detect run length L in [1..min_len-1]
        # and penalize (min_len - L) if run ends (or horizon ends) after L ones.
        if min_len <= 1:
            return

        # Create start indicators for runs of ones in the planning horizon, considering historical prefix.
        starts = []
        for i in range(len(seq_vars)):
            st = model.NewIntVar(0, 1, f"{name_prefix}_start[{i}]")
            if i == 0:
                if hist_ones > 0:
                    # previous day was 1, so cannot start at i=0
                    model.Add(st == 0)
                else:
                    # start if seq[0]==1
                    model.Add(st <= seq_vars[0])
                    model.Add(st >= seq_vars[0])
            else:
                # st = 1 if seq[i]=1 and seq[i-1]=0
                model.Add(st <= seq_vars[i])
                model.Add(st <= 1 - seq_vars[i - 1])
                model.Add(st >= seq_vars[i] - seq_vars[i - 1])
            starts.append(st)

        # For each start, compute run length up to min_len-1 and whether it ends within that range.
        for i, st in enumerate(starts):
            for L in range(1, min_len):
                # Conditions for a run of exactly length L starting at i:
                # st=1, seq[i..i+L-1]=1, and (i+L==len or seq[i+L]=0)
                if i + L - 1 >= len(seq_vars):
                    continue
                all_ones = seq_vars[i : i + L]
                end_is_zero = None
                if i + L < len(seq_vars):
                    end_is_zero = model.NewIntVar(0, 1, f"{name_prefix}_end0[{i},{L}]")
                    model.Add(end_is_zero == 1 - seq_vars[i + L])
                else:
                    # horizon ends; treat as end of run
                    end_is_zero = None

                exact = model.NewIntVar(0, 1, f"{name_prefix}_run_exact[{i},{L}]")
                # exact <= st and <= each one and <= end_is_zero (if exists)
                model.Add(exact <= st)
                for v in all_ones:
                    model.Add(exact <= v)
                if end_is_zero is not None:
                    model.Add(exact <= end_is_zero)

                # Lower bound: if all conditions true then exact=1
                # exact >= st + sum(all_ones) + end_is_zero - (L + 1)  (or without end)
                if end_is_zero is not None:
                    model.Add(exact >= st + sum(all_ones) + end_is_zero - (L + 1))
                else:
                    model.Add(exact >= st + sum(all_ones) - (L))
                missing = min_len - L
                if missing > 0:
                    penalty_terms.append(weight * missing * exact)

        # Historical run that continues into horizon and ends early: if hist_ones>0 and first days are ones then end.
        if hist_ones > 0:
            for L in range(0, min_len):
                # total run length = hist_ones + L, where L is number of initial ones in horizon before first zero (or end)
                # If hist_ones already >= min_len, no missing.
                if hist_ones + L >= min_len:
                    continue
                if L > len(seq_vars):
                    continue
                # Need: first L are ones, and next is zero (or end), and also seq[0] must be 1 if L>0, else seq[0]=0.
                cond = model.NewIntVar(0, 1, f"{name_prefix}_hist_end[{L}]")
                if L == 0:
                    # ends immediately if first is 0 (or horizon empty)
                    if len(seq_vars) == 0:
                        model.Add(cond == 1)
                    else:
                        model.Add(cond == 1 - seq_vars[0])
                else:
                    # first L are ones
                    for j in range(L):
                        model.Add(cond <= seq_vars[j])
                    # end at next is zero or horizon end
                    if L < len(seq_vars):
                        end0 = model.NewIntVar(0, 1, f"{name_prefix}_hist_end0[{L}]")
                        model.Add(end0 == 1 - seq_vars[L])
                        model.Add(cond <= end0)
                        model.Add(cond >= sum(seq_vars[:L]) + end0 - L)
                    else:
                        model.Add(cond >= sum(seq_vars[:L]) - L + 1)
                missing = min_len - (hist_ones + L)
                if missing > 0:
                    penalty_terms.append(weight * missing * cond)

    # S2a. Consecutive shift assignments per shift type (min/max per shift)
    for n in nurses:
        hist_last_shift = safe_get(historical, n, "lastAssignedShiftType", default="None")
        hist_consec_assign = int(safe_get(historical, n, "numberOfConsecutiveAssignments", default=0) or 0)
        for s in shifts:
            min_c = int(safe_get(shift_type_limits, s, "min_consec", default=0) or 0)
            max_c = int(safe_get(shift_type_limits, s, "max_consec", default=len(days)) or len(days))
            seq = [shift_work[(n, d, s)] for d in days]
            hist_ones = hist_consec_assign if hist_last_shift == s else 0
            # hist_zeros not used in this function; pass 0
            add_consecutive_penalties_binary(seq, hist_ones=hist_ones, hist_zeros=0, min_len=min_c, max_len=max_c,
                                            weight=w_consec_shift, name_prefix=f"consec_shift[{n},{s}]")

    # S2b. Consecutive working days (contract min/max)
    for n in nurses:
        c = contracts.get(n, {})
        min_cw = int(c.get("min_consec_work", 0) or 0)
        max_cw = int(c.get("max_consec_work", len(days)) or len(days))
        hist_cw = int(safe_get(historical, n, "numberOfConsecutiveWorkingDays", default=0) or 0)
        seq = [work[(n, d)] for d in days]
        add_consecutive_penalties_binary(seq, hist_ones=hist_cw, hist_zeros=0, min_len=min_cw, max_len=max_cw,
                                        weight=w_consec_work, name_prefix=f"consec_work[{n}]")

    # S3. Consecutive days off (contract min/max)
    for n in nurses:
        c = contracts.get(n, {})
        min_co = int(c.get("min_consec_off", 0) or 0)
        max_co = int(c.get("max_consec_off", len(days)) or len(days))
        hist_co = int(safe_get(historical, n, "numberOfConsecutiveDaysOff", default=0) or 0)
        seq = [off[(n, d)] for d in days]
        add_consecutive_penalties_binary(seq, hist_ones=hist_co, hist_zeros=0, min_len=min_co, max_len=max_co,
                                        weight=w_consec_off, name_prefix=f"consec_off[{n}]")

    # S4. Preferences (undesired assignments)
    # Preferences are shift-OFF requests.
    pref_map_any = defaultdict(set)  # (n,d) -> True
    pref_map_shift = defaultdict(set)  # (n,d) -> set(shifts)
    for p in preferences:
        n = p.get("nurse")
        d = p.get("day")
        sh = p.get("shift")
        if n not in nurses or d not in days:
            continue
        if sh == "Any":
            pref_map_any[(n, d)].add("Any")
        elif sh in shifts:
            pref_map_shift[(n, d)].add(sh)
        else:
            # Unknown shift label in preference; cannot model exactly.
            # Kept as comment per requirements.
            pass

    for (n, d), _ in pref_map_any.items():
        # penalty if assigned to any shift that day
        pen = model.NewIntVar(0, 1, f"pref_any[{n},{d}]")
        model.Add(pen == work[(n, d)])
        penalty_terms.append(w_prefs * pen)

    for (n, d), shset in pref_map_shift.items():
        for s in shset:
            pen = model.NewIntVar(0, 1, f"pref_shift[{n},{d},{s}]")
            model.Add(pen == shift_work[(n, d, s)])
            penalty_terms.append(w_prefs * pen)

    # S5. Complete weekend
    for n in nurses:
        c = contracts.get(n, {})
        if int(c.get("complete_weekend", 0) or 0) != 1:
            continue
        for w in range(num_weeks):
            if w not in week_to_sat or w not in week_to_sun:
                continue
            sat = week_to_sat[w]
            sun = week_to_sun[w]
            satw = work[(n, sat)]
            sunw = work[(n, sun)]
            diff = model.NewIntVar(0, 1, f"weekend_diff[{n},W{w+1}]")
            model.Add(diff >= satw - sunw)
            model.Add(diff >= sunw - satw)
            penalty_terms.append(w_complete_weekend * diff)

    # S6. Total assignments (working days) deviation from min/max
    for n in nurses:
        c = contracts.get(n, {})
        min_total = int(c.get("min_total", 0) or 0)
        max_total = int(c.get("max_total", len(days)) or len(days))
        hist_total = int(safe_get(historical, n, "numberOfAssignments", default=0) or 0)
        total_work = model.NewIntVar(0, len(days), f"total_work[{n}]")
        model.Add(total_work == sum(work[(n, d)] for d in days))
        total_with_hist = model.NewIntVar(0, len(days) + hist_total, f"total_with_hist[{n}]")
        model.Add(total_with_hist == total_work + hist_total)

        under = model.NewIntVar(0, len(days) + hist_total, f"total_under[{n}]")
        over = model.NewIntVar(0, len(days) + hist_total, f"total_over[{n}]")
        model.Add(under >= min_total - total_with_hist)
        model.Add(under >= 0)
        model.Add(over >= total_with_hist - max_total)
        model.Add(over >= 0)
        penalty_terms.append(w_total_assign * under)
        penalty_terms.append(w_total_assign * over)

    # S7. Total working weekends excess above max_weekends
    for n in nurses:
        c = contracts.get(n, {})
        max_weekends = int(c.get("max_weekends", 0) or 0)
        hist_weekends = int(safe_get(historical, n, "numberOfWorkingWeekends", default=0) or 0)

        worked_weekend_vars = []
        for w in range(num_weeks):
            if w not in week_to_sat or w not in week_to_sun:
                continue
            sat = week_to_sat[w]
            sun = week_to_sun[w]
            ww = model.NewIntVar(0, 1, f"worked_weekend[{n},W{w+1}]")
            model.Add(ww >= work[(n, sat)])
            model.Add(ww >= work[(n, sun)])
            model.Add(ww <= work[(n, sat)] + work[(n, sun)])
            worked_weekend_vars.append(ww)

        total_weekends = model.NewIntVar(0, len(worked_weekend_vars), f"total_weekends[{n}]")
        model.Add(total_weekends == sum(worked_weekend_vars) if worked_weekend_vars else 0)

        total_weekends_with_hist = model.NewIntVar(0, len(worked_weekend_vars) + hist_weekends, f"total_weekends_with_hist[{n}]")
        model.Add(total_weekends_with_hist == total_weekends + hist_weekends)

        excess = model.NewIntVar(0, len(worked_weekend_vars) + hist_weekends, f"weekends_excess[{n}]")
        model.Add(excess >= total_weekends_with_hist - max_weekends)
        model.Add(excess >= 0)
        penalty_terms.append(w_total_weekends * excess)

    # Objective
    model.Minimize(sum(penalty_terms) if penalty_terms else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No feasible solution")
        return

    # H3 Internal Post-Solve Audit
    # Must validate forbidden successions including historical boundary.
    def val(var):
        return int(solver.Value(var))

    for n in nurses:
        for di in range(len(days) - 1):
            d_prev = days[di]
            d_curr = days[di + 1]
            for a, b in forbidden_pairs:
                if val(shift_work[(n, d_prev, a)]) + val(shift_work[(n, d_curr, b)]) > 1:
                    print("Solver failed")
                    return

    for n in nurses:
        first_day = days[0]
        last_shift = safe_get(historical, n, "lastAssignedShiftType", default="None")
        if last_shift is None:
            last_shift = "None"
        if last_shift != "None":
            for a, b in forbidden_pairs:
                if a == last_shift:
                    if val(shift_work[(n, first_day, b)]) != 0:
                        print("Solver failed")
                        return

    # INRC2 Weekly Solution Export
    os.makedirs("initial_roster", exist_ok=True)

    shift_map = {"E": "Early", "L": "Late", "N": "Night"}

    weekly_assignments = {w: [] for w in range(num_weeks)}
    for n in nurses:
        for d in days:
            wi = day_to_week[d]
            dow = day_to_dow[d]
            for s in shifts:
                for k in skills:
                    if val(x[(n, d, s, k)]) == 1:
                        weekly_assignments[wi].append(
                            {
                                "nurse": n,
                                "day": dow,
                                "shiftType": shift_map.get(s, s),  # if unknown, keep original
                                "skill": k,
                            }
                        )

    def sort_key(a):
        return (a["nurse"], dow_order.get(a["day"], 99), a["shiftType"], a["skill"])

    for w in range(num_weeks):
        weekly_assignments[w].sort(key=sort_key)
        out_obj = {"scenario": scenario_name, "week": w, "assignments": weekly_assignments[w]}
        out_path = os.path.join("initial_roster", f"week{w}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print("Done")


if __name__ == "__main__":
    main()