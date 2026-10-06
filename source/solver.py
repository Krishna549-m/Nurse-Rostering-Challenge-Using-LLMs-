import json
import os
from ortools.sat.python import cp_model

# Load input data
with open("response1.json", "r") as f:
    data = json.load(f)

sets = data["sets"]
params = data["parameters"]

N = sets["N"]
D = sets["D"]
S = sets["S"]
K = sets["K"]

coverage = params["coverage"]
qualifications = params["qualifications"]
contracts = params["contracts"]
shift_type_limits = params["shift_type_limits"]
shift_successions = params["shift_successions"]
historical = params["historical_assignments"]
preferences = params["preferences"]
weights = params["soft_rule_weights"]

forbidden_pairs = [tuple(p) for p in shift_successions["forbidden"]]

# Scenario name: no explicit scenario field in input; use documented fallback
scenario_name = data.get("scenario", "unknown_scenario")

# Shift name mapping for INRC2 export
shift_name_map = {"E": "Early", "L": "Late", "N": "Night"}
day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Parse day labels into (week_index, day_name)
def parse_day(label):
    # label like "W1_Mon"
    wk_part, dname = label.split("_")
    wk = int(wk_part[1:]) - 1  # zero-based export numbering
    return wk, dname

day_info = {d: parse_day(d) for d in D}

model = cp_model.CpModel()

# Primary decision variables x[n,d,s,k]
x = {}
for n in N:
    for d in D:
        for s in S:
            for k in K:
                x[(n, d, s, k)] = model.NewIntVar(0, 1, f"x_{n}_{d}_{s}_{k}")

# Linking variables
# shift_work[n,d,s] = sum_k x[n,d,s,k]
shift_work = {}
for n in N:
    for d in D:
        for s in S:
            sw = model.NewIntVar(0, 1, f"sw_{n}_{d}_{s}")
            model.Add(sw == sum(x[(n, d, s, k)] for k in K))
            shift_work[(n, d, s)] = sw

# work[n,d] = sum_s shift_work[n,d,s]
work = {}
off = {}
for n in N:
    for d in D:
        w = model.NewIntVar(0, 1, f"work_{n}_{d}")
        model.Add(w == sum(shift_work[(n, d, s)] for s in S))
        work[(n, d)] = w
        o = model.NewIntVar(0, 1, f"off_{n}_{d}")
        model.Add(o == 1 - w)
        off[(n, d)] = o

# ---------------------------------------------------------------
# HARD CONSTRAINTS
# ---------------------------------------------------------------

# H1. Single assignment per day
for n in N:
    for d in D:
        model.Add(sum(x[(n, d, s, k)] for s in S for k in K) <= 1)

# H2. Minimum coverage
for d in D:
    for s in S:
        for k in K:
            mincov = coverage[d][s][k]["min"]
            model.Add(sum(x[(n, d, s, k)] for n in N) >= mincov)

# H4. Skill compatibility
for n in N:
    for k in K:
        qual = qualifications[n].get(k, 0)
        if qual == 0:
            for d in D:
                for s in S:
                    model.Add(x[(n, d, s, k)] == 0)

# H3 Shift Successions (Hard Constraint)
for n in N:
    for di in range(len(D) - 1):
        d_prev = D[di]
        d_curr = D[di + 1]
        for a, b in forbidden_pairs:
            model.Add(shift_work[(n, d_prev, a)] + shift_work[(n, d_curr, b)] <= 1)

for n in N:
    first_day = D[0]
    last_shift = historical[n]["lastAssignedShiftType"]
    if last_shift != "None":
        for a, b in forbidden_pairs:
            if a == last_shift:
                model.Add(shift_work[(n, first_day, b)] == 0)

# ---------------------------------------------------------------
# SOFT CONSTRAINTS (penalties)
# ---------------------------------------------------------------
penalty_terms = []

W_OPT = weights["optimal_coverage"]
W_PREF = weights["preferences"]
W_CONSEC_SHIFT = weights["consecutive_shift_assignments"]
W_CONSEC_WORK = weights["consecutive_working_days"]
W_CONSEC_OFF = weights["consecutive_days_off"]
W_WEEKEND = weights["complete_weekend"]
W_TOTAL = weights["total_assignments"]
W_TOTAL_WKND = weights["total_working_weekends"]

# S1. Optimal coverage (shortage below opt penalized)
for d in D:
    for s in S:
        for k in K:
            opt = coverage[d][s][k]["opt"]
            if opt > 0:
                assigned = sum(x[(n, d, s, k)] for n in N)
                short = model.NewIntVar(0, opt, f"short_{d}_{s}_{k}")
                # short >= opt - assigned ; short >= 0
                model.Add(short >= opt - assigned)
                penalty_terms.append(W_OPT * short)

# S4. Preferences (undesired assignments)
for pref in preferences:
    n = pref["nurse"]
    d = pref["day"]
    sh = pref["shift"]
    if n not in N or d not in D:
        continue
    if sh == "Any":
        # penalty if works any shift that day
        penalty_terms.append(W_PREF * work[(n, d)])
    else:
        # penalty if assigned that specific shift
        penalty_terms.append(W_PREF * shift_work[(n, d, sh)])

# S5. Complete weekend: penalize working exactly one of Sat/Sun
# Group days by (week, is Sat/Sun)
week_indices = sorted(set(wk for (wk, dn) in day_info.values()))
sat_by_week = {}
sun_by_week = {}
for d in D:
    wk, dn = day_info[d]
    if dn == "Sat":
        sat_by_week[wk] = d
    elif dn == "Sun":
        sun_by_week[wk] = d

for n in N:
    if contracts[n]["complete_weekend"] == 1:
        for wk in week_indices:
            if wk in sat_by_week and wk in sun_by_week:
                sat_d = sat_by_week[wk]
                sun_d = sun_by_week[wk]
                # diff = |work_sat - work_sun| ; penalize = 1 if exactly one worked
                mism = model.NewIntVar(0, 1, f"wkndmis_{n}_{wk}")
                model.Add(mism >= work[(n, sat_d)] - work[(n, sun_d)])
                model.Add(mism >= work[(n, sun_d)] - work[(n, sat_d)])
                penalty_terms.append(W_WEEKEND * mism)

# S6. Total assignments deviation
num_days = len(D)
for n in N:
    total = model.NewIntVar(0, num_days, f"total_{n}")
    model.Add(total == sum(work[(n, d)] for d in D))
    min_total = contracts[n]["min_total"]
    max_total = contracts[n]["max_total"]
    under = model.NewIntVar(0, num_days, f"undertot_{n}")
    over = model.NewIntVar(0, num_days, f"overtot_{n}")
    model.Add(under >= min_total - total)
    model.Add(over >= total - max_total)
    penalty_terms.append(W_TOTAL * under)
    penalty_terms.append(W_TOTAL * over)

# S7. Total working weekends: excess above max_weekends
for n in N:
    max_wknd = contracts[n]["max_weekends"]
    wknd_worked_vars = []
    for wk in week_indices:
        if wk in sat_by_week or wk in sun_by_week:
            ww = model.NewIntVar(0, 1, f"wkndworked_{n}_{wk}")
            terms = []
            if wk in sat_by_week:
                terms.append(work[(n, sat_by_week[wk])])
            if wk in sun_by_week:
                terms.append(work[(n, sun_by_week[wk])])
            # ww = 1 if at least one worked
            for t in terms:
                model.Add(ww >= t)
            model.Add(ww <= sum(terms))
            wknd_worked_vars.append(ww)
    if wknd_worked_vars:
        total_wknd = model.NewIntVar(0, len(wknd_worked_vars), f"totalwknd_{n}")
        model.Add(total_wknd == sum(wknd_worked_vars))
        excess = model.NewIntVar(0, len(wknd_worked_vars), f"wkndexcess_{n}")
        model.Add(excess >= total_wknd - max_wknd)
        penalty_terms.append(W_TOTAL_WKND * excess)

# ---------------------------------------------------------------
# S2b. Consecutive working days & S3 consecutive days off (with history)
# ---------------------------------------------------------------
# We build penalties using a sequence-based approach with historical boundary.
# For consecutive working days: penalize violation of min/max consec_work.
# For consecutive days off: penalize violation of min/max consec_off.
#
# Approach: For each nurse we create an extended boolean sequence.
# History gives numberOfConsecutiveWorkingDays / numberOfConsecutiveDaysOff prior
# to the first planning day. We model min/max via block-window constraints as soft.
#
# MAX consecutive working days (soft): for any window of length (max+1) worked days,
# penalize. We linearize per window using a sum.
# MIN consecutive working days (soft): if a work block starts (transition off->work),
# penalize if it is shorter than min. This is handled with start indicators and
# lookahead; historical boundary handled by treating pre-first-day state.

def add_consec_penalties(kind):
    # kind: 'work' uses work[] and contract min/max_consec_work, weight W_CONSEC_WORK
    #       'off'  uses off[] and contract min/max_consec_off, weight W_CONSEC_OFF
    for n in N:
        if kind == "work":
            var = work
            mn = contracts[n]["min_consec_work"]
            mx = contracts[n]["max_consec_work"]
            hist = historical[n]["numberOfConsecutiveWorkingDays"]
            wgt = W_CONSEC_WORK
            state_before = 1 if hist > 0 else 0
        else:
            var = off
            mn = contracts[n]["min_consec_off"]
            mx = contracts[n]["max_consec_off"]
            hist = historical[n]["numberOfConsecutiveDaysOff"]
            wgt = W_CONSEC_OFF
            state_before = 1 if hist > 0 else 0

        seq = [var[(n, d)] for d in D]
        L = len(seq)

        # MAX constraint (soft): any (mx+1) consecutive of same state -> penalize excess.
        # Include historical carry: if state_before active with 'hist' count, then a
        # run continuing into planning may exceed mx sooner.
        # Window over planning days:
        for start in range(0, L - mx):
            window = seq[start:start + mx + 1]
            over = model.NewIntVar(0, 1, f"cmax_{kind}_{n}_{start}")
            # if all mx+1 are same-state -> sum == mx+1 -> over should be 1
            model.Add(over >= sum(window) - mx)
            penalty_terms.append(wgt * over)

        # Historical max: prefix window using state_before repeated 'hist' times
        if state_before == 1 and hist > 0:
            # remaining allowance before hitting max
            allow = mx - hist
            if allow < 0:
                allow = 0
            # window of planning days of length (allow+1) starting at day0 that continue run
            wlen = allow + 1
            if wlen <= L:
                window = seq[0:wlen]
                over = model.NewIntVar(0, 1, f"cmaxhist_{kind}_{n}")
                model.Add(over >= sum(window) - allow)
                penalty_terms.append(wgt * over)

        # MIN constraint (soft): a block start that yields a run shorter than mn.
        # start[i] = var[i]==1 and (var[i-1]==0 or i==0 with state_before==0)
        for i in range(L):
            si = model.NewIntVar(0, 1, f"start_{kind}_{n}_{i}")
            if i == 0:
                if state_before == 0:
                    # start iff seq[0]==1
                    model.Add(si == seq[0])
                else:
                    # run continues from history; not a fresh start under-min risk
                    model.Add(si == 0)
            else:
                # si >= seq[i] - seq[i-1]; si <= seq[i]; si <= 1-seq[i-1]
                model.Add(si <= seq[i])
                model.Add(si <= 1 - seq[i - 1])
                model.Add(si >= seq[i] - seq[i - 1])
            # If start at i, require next mn days (including i) all same-state else penalize
            # shortfall = mn - (number of consecutive same-state from i) when start
            need = mn
            avail = seq[i:i + need]
            # count of same-state in the needed window
            cnt = model.NewIntVar(0, need, f"cnt_{kind}_{n}_{i}")
            model.Add(cnt == sum(avail))
            # shortfall only relevant if si==1
            short = model.NewIntVar(0, need, f"minshort_{kind}_{n}_{i}")
            # short >= (need - cnt) - (1-si)*need
            model.Add(short >= (need - cnt) - (1 - si) * need)
            penalty_terms.append(wgt * short)

add_consec_penalties("work")
add_consec_penalties("off")

# ---------------------------------------------------------------
# S2a. Consecutive shift assignments per shift type (with history), weight 15
# ---------------------------------------------------------------
for n in N:
    for s in S:
        mn = shift_type_limits[s]["min_consec"]
        mx = shift_type_limits[s]["max_consec"]
        seq = [shift_work[(n, d, s)] for d in D]
        L = len(seq)
        last_shift = historical[n]["lastAssignedShiftType"]
        hist_cnt = historical[n]["numberOfConsecutiveAssignments"] if last_shift == s else 0
        state_before = 1 if hist_cnt > 0 else 0

        # MAX
        for start in range(0, L - mx):
            window = seq[start:start + mx + 1]
            over = model.NewIntVar(0, 1, f"smax_{n}_{s}_{start}")
            model.Add(over >= sum(window) - mx)
            penalty_terms.append(W_CONSEC_SHIFT * over)

        if state_before == 1 and hist_cnt > 0:
            allow = mx - hist_cnt
            if allow < 0:
                allow = 0
            wlen = allow + 1
            if wlen <= L:
                window = seq[0:wlen]
                over = model.NewIntVar(0, 1, f"smaxhist_{n}_{s}")
                model.Add(over >= sum(window) - allow)
                penalty_terms.append(W_CONSEC_SHIFT * over)

        # MIN
        for i in range(L):
            si = model.NewIntVar(0, 1, f"sstart_{n}_{s}_{i}")
            if i == 0:
                if state_before == 0:
                    model.Add(si == seq[0])
                else:
                    model.Add(si == 0)
            else:
                model.Add(si <= seq[i])
                model.Add(si <= 1 - seq[i - 1])
                model.Add(si >= seq[i] - seq[i - 1])
            need = mn
            avail = seq[i:i + need]
            cnt = model.NewIntVar(0, need, f"scnt_{n}_{s}_{i}")
            model.Add(cnt == sum(avail))
            short = model.NewIntVar(0, need, f"sminshort_{n}_{s}_{i}")
            model.Add(short >= (need - cnt) - (1 - si) * need)
            penalty_terms.append(W_CONSEC_SHIFT * short)

# ---------------------------------------------------------------
# OBJECTIVE
# ---------------------------------------------------------------
model.Minimize(sum(penalty_terms))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60.0
solver.parameters.num_search_workers = 8

status = solver.Solve(model)

if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("No feasible solution")
else:
    # H3 Internal Post-Solve Audit
    audit_ok = True
    for n in N:
        for di in range(len(D) - 1):
            d_prev = D[di]
            d_curr = D[di + 1]
            for a, b in forbidden_pairs:
                v_prev = solver.Value(shift_work[(n, d_prev, a)])
                v_curr = solver.Value(shift_work[(n, d_curr, b)])
                if v_prev + v_curr > 1:
                    audit_ok = False
        first_day = D[0]
        last_shift = historical[n]["lastAssignedShiftType"]
        if last_shift != "None":
            for a, b in forbidden_pairs:
                if a == last_shift:
                    if solver.Value(shift_work[(n, first_day, b)]) != 0:
                        audit_ok = False

    if not audit_ok:
        print("Solver failed")
    else:
        # INRC2 Weekly Solution Export
        os.makedirs("initial_roster", exist_ok=True)

        weekly_assignments = {wk: [] for wk in week_indices}
        for n in N:
            for d in D:
                wk, dn = day_info[d]
                for s in S:
                    for k in K:
                        if solver.Value(x[(n, d, s, k)]) == 1:
                            weekly_assignments[wk].append({
                                "nurse": n,
                                "day": dn,
                                "shiftType": shift_name_map[s],
                                "skill": k
                            })

        day_rank = {dn: i for i, dn in enumerate(day_order)}
        shift_rank = {"Early": 0, "Late": 1, "Night": 2}

        for wk in week_indices:
            assigns = weekly_assignments[wk]
            assigns.sort(key=lambda a: (a["nurse"], day_rank.get(a["day"], 99),
                                        shift_rank.get(a["shiftType"], 99)))
            sol = {
                "scenario": scenario_name,
                "week": wk,
                "assignments": assigns
            }
            with open(os.path.join("initial_roster", f"week{wk}.json"), "w") as f:
                json.dump(sol, f, indent=2)

        print("Solver finished")