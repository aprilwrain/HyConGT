from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import warnings

import numpy as np
import pandas as pd

import model_params as P

warnings.filterwarnings("ignore")


# =====================================================================
# 0. I/O helpers
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent


def first_existing(candidates):
    for p in candidates:
        pp = Path(p)
        if not pp.is_absolute():
            pp = BASE_DIR / pp
        if pp.exists():
            return pp
    raise FileNotFoundError("Cannot locate input file. Tried:\n" +
                            "\n".join(map(str, candidates)))


def read_csv_auto(path):
    """Read CSV trying several encodings (utf-8-sig is preferred)."""
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def numeric(row, names, default=0.0):
    """Look up a numeric attribute by trying several column names."""
    for name in names:
        if name in row.index and pd.notna(row[name]):
            try:
                return float(row[name])
            except (TypeError, ValueError):
                return default
    return default


# =====================================================================
# 1. Inputs
# =====================================================================
NODES_PATH = first_existing(["data/node.csv"])
EDGES_PATH = first_existing(["data/edges.csv"])
SUB_PATH   = first_existing(["data/sub.csv"])

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE = OUT_DIR / "node_outflow_physcal.csv"

nodes_df = read_csv_auto(NODES_PATH)
edges_df = read_csv_auto(EDGES_PATH)
sub_df   = read_csv_auto(SUB_PATH)

# ------ Edge weights --------------------------------------------------
edge_weight_col = ("weight_alpha" if "weight_alpha" in edges_df.columns
                   else "ratio"   if "ratio"        in edges_df.columns
                   else None)
if edge_weight_col is None:
    raise ValueError("edges file must contain 'weight_alpha' or 'ratio'")

split = {(r["from_id"], r["to_id"]): float(r[edge_weight_col])
         for _, r in edges_df.iterrows()}
for k, v in P.DEFAULT_SPLITS.items():
    split.setdefault(k, v)


# =====================================================================
# 2. SWAT daily hydrology pivoted by sub-basin
# =====================================================================
def parse_sub_dates(df):
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "YYYYDDD" in df.columns:
        df["date"] = pd.to_datetime(df["YYYYDDD"].astype(str), format="%Y%j")
    elif {"YEAR", "DAY"}.issubset(df.columns):
        df["date"] = [date(int(y), 1, 1) + timedelta(days=int(d) - 1)
                      for y, d in zip(df["YEAR"], df["DAY"])]
        df["date"] = pd.to_datetime(df["date"])
    elif {"YEAR", "MON"}.issubset(df.columns) \
         and pd.to_numeric(df["MON"], errors="coerce").max() > 12:
        df["date"] = [date(int(y), 1, 1) + timedelta(days=int(d) - 1)
                      for y, d in zip(df["YEAR"], df["MON"])]
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Cannot infer dates from SWAT sub file.")
    df["SUB"] = pd.to_numeric(df["SUB"], errors="coerce").astype(int)
    return df.sort_values(["date", "SUB"])


sub_df = parse_sub_dates(sub_df)
for c in P.HYDRO_COLS:
    if c not in sub_df.columns:
        sub_df[c] = 0.0
    sub_df[c] = pd.to_numeric(sub_df[c], errors="coerce").fillna(0.0)

all_dates = pd.DatetimeIndex(sorted(sub_df["date"].unique()))


def pivot_var(name):
    piv = (sub_df.pivot_table(index="date", columns="SUB", values=name,
                              aggfunc="mean")
                 .reindex(all_dates).sort_index().fillna(0.0))
    piv.columns = [int(c) for c in piv.columns]
    return piv


PRECIP = pivot_var("PRECIPmm")
WYLD   = pivot_var("WYLDmm")
ETmm   = pivot_var("ETmm")


def sub_series(piv, sub):
    return piv[sub].values.astype(float) if sub in piv.columns \
           else np.zeros(len(all_dates), dtype=float)


# =====================================================================
# 3. Static engineering attributes from node2.csv
# =====================================================================
def lookup_node(node_id):
    rows = nodes_df.loc[nodes_df["node_id"].astype(str) == node_id]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=float)


def node_attr(node_id, names, default):
    row = lookup_node(node_id)
    return numeric(row, names, default) if not row.empty else default


# ---- Source nodes ----
SOURCES = {}
for _, r in nodes_df[nodes_df["node_type"].astype(str).str.upper() == "S"
                     ].iterrows():
    nid = str(r["node_id"])
    SOURCES[nid] = {
        "sub":   int(numeric(r, ["SUB"], 0)),
        "area":  numeric(r, ["jiangyuhuishuimianji_m2",
                             "catchment_area_m2"], 0.0),
        "rc":    numeric(r, ["jingliuxishu", "runoff_coef"], 0.45),
        "kind":  P.SOURCE_KIND.get(nid, "dump"),
    }

# ---- Collection ponds (R1, R2, R3) ----
POND_IDS = ["R1", "R2", "R3"]
PONDS = {}
for nid in POND_IDS:
    v_max = node_attr(nid, ["youxiaokurong_m3"], 1.0e6) / 1e4   # 万m3
    q_max = node_attr(nid, ["zuidabengpainengli_m3/h"], 1000.0) \
            * 24.0 * P.POND_PUMP_DUTY_CYCLE / 1e4               # 万m3/d
    PONDS[nid] = {"v_max":  v_max,
                  "q_max":  q_max,
                  "v_dead": v_max * P.POND_DEAD_STORAGE_FRAC,
                  "v0":     v_max * P.POND_INITIAL_FILL_FRAC}

# ---- Tailings storage facilities (R6, R7) ----
TSF_IDS = ["R6", "R7"]
TSF = {}
for nid in TSF_IDS:
    r = lookup_node(nid)
    TSF[nid] = {
        "sub":        int(numeric(r, ["SUB"], 34 if nid == "R6" else 38)),
        "area_catch": numeric(r, ["jiangyuhuishuimianji_m2"],
                              7.8e6 if nid == "R6" else 1.541e7),
        "area_pond":  numeric(r, ["shuimianmianji_m2"],
                              4.0e6 if nid == "R6" else 6.1e5),
        "rc":         max(P.TSF_CATCHMENT_RC_FLOOR,
                          numeric(r, ["jingliuxishu"], 0.40)),
        "slurry":     P.TSF_SLURRY_INFLOW[nid],
        "decant_k":   P.TSF_DECANT_K[nid],
        "decant_min": P.TSF_DECANT_MIN[nid],
        "seep_base":  P.TSF_SEEPAGE_BASE[nid],
        "seep_k":     P.TSF_SEEPAGE_K[nid],
        "v0":         P.TSF_INITIAL_STORAGE[nid],
    }

# ---- WWTPs (T1, T2) ----
WWTP_IDS = ["T1", "T2"]
WWTP = {}
for nid in WWTP_IDS:
    WWTP[nid] = {
        "sub":      int(node_attr(nid, ["SUB"], 44 if nid == "T1" else 36)),
        "q_design": node_attr(nid, ["zuidabengpainengli_m3/h"], 3200.0)
                    * 24.0 / 1e4,                            # 万m3/d
        "hrt":      P.WWTP_HRT_DAYS[nid],
        "base":     P.WWTP_BASEFLOW[nid],
    }


# =====================================================================
# 4. Physical operators
# =====================================================================
def linear_reservoir(inflow, k_days, storage_0=0.0):
    n = len(inflow)
    out = np.zeros(n, dtype=float)
    s = float(storage_0)
    alpha = 1.0 - np.exp(-1.0 / max(k_days, 1.0e-3))
    for i in range(n):
        s += float(inflow[i])
        q  = alpha * s
        out[i] = q
        s -= q
    return out


def nash_cascade(inflow, n_reservoirs, k_days):
    flow = np.asarray(inflow, dtype=float)
    for _ in range(int(n_reservoirs)):
        flow = linear_reservoir(flow, k_days)
    return flow


def source_flow_series(s_id):
    p = SOURCES[s_id]
    sub_id = p["sub"]
    area   = p["area"]                                  # m^2
    rc     = p["rc"]
    yld    = sub_series(WYLD, sub_id)                   # mm/d

    # (a) Surface runoff routed through the Nash cascade
    q_surf = rc * area * yld / 1e7                      # 万m3/d, undelayed
    q_routed = nash_cascade(q_surf,
                            P.SOURCE_NASH_N,
                            P.SOURCE_NASH_K_DAYS)

    # (b) Slow pore-water drainage
    if p["kind"] == "pit":
        perc_frac, k_drain = P.PIT_PERC_FRACTION, P.PIT_DRAINAGE_K
    else:
        perc_frac, k_drain = (P.WASTE_ROCK_PERC_FRACTION,
                              P.WASTE_ROCK_DRAINAGE_K)
    percolation = perc_frac * area * yld / 1e7
    q_delayed   = linear_reservoir(percolation, k_drain)

    return q_routed + q_delayed


def pond_step(v_prev, q_in, pond):
    v_new   = v_prev + q_in
    q_pump  = min(max(0.0, v_new - pond["v_dead"]), pond["q_max"])
    v_after = v_new - q_pump
    q_spill = max(0.0, v_after - pond["v_max"])
    v_after = min(v_after, pond["v_max"])
    return q_pump, q_spill, max(0.0, v_after)


def tsf_step(v_prev, q_process_in, idx, t, rain_routed_series, decant_prev):
    v = v_prev + t["slurry"] + q_process_in + float(rain_routed_series[idx])
    v = max(0.0, v)

    # Darcy seepage (storage-proportional + baseline)
    q_seep = t["seep_base"] + t["seep_k"] * v
    q_seep = min(q_seep, v)
    v -= q_seep

    # Storage-based target rate, with a process-water floor
    q_target = max(t["decant_min"], t["decant_k"] * v)

    # First-order operator response (low-pass on the pump set-point)
    beta = 1.0 - np.exp(-1.0 / max(P.TSF_OPERATOR_RESPONSE_DAYS, 1.0e-3))
    q_decant = (1.0 - beta) * decant_prev + beta * q_target
    q_decant = min(q_decant, v)
    v -= q_decant

    return ({"v": v, "decant": q_decant, "seep": q_seep}, q_decant)


def wwtp_step(q_in, q_out_prev, plant):
    alpha    = 1.0 - np.exp(-1.0 / max(plant["hrt"], 1.0e-3))
    q_mixed  = (1.0 - alpha) * q_out_prev + alpha * q_in
    q_bypass = max(0.0, q_in - plant["q_design"])
    q_out    = min(q_mixed, plant["q_design"])
    return q_out, q_bypass


# =====================================================================
# 5. Pre-compute source-node series and T5 driving series
# =====================================================================
S_FLOW = {sid: source_flow_series(sid) for sid in SOURCES}
TSF_RAIN_ROUTED = {}
for _nid in TSF_IDS:
    _t = TSF[_nid]
    _p = sub_series(PRECIP, _t["sub"])
    _e = sub_series(ETmm,   _t["sub"])
    _y = sub_series(WYLD,   _t["sub"])
    _rain_in = (_p * _t["area_pond"]                             # rain on pond
                + _t["rc"] * max(0.0, _t["area_catch"] - _t["area_pond"])
                  * _y                                           # perimeter runoff
                - _e * _t["area_pond"]) / 1e7                    # ET from pond
    TSF_RAIN_ROUTED[_nid] = nash_cascade(_rain_in,
                                         P.TSF_RAIN_NASH_N,
                                         P.TSF_RAIN_NASH_K_DAYS)

SUB_T5 = int(node_attr("T5", ["SUB"], 42))
T5_local_runoff = sub_series(WYLD, SUB_T5) * P.T5_LOCAL_AREA_KM2 * 0.1
T5_SERIES = P.WWTP_BASEFLOW["T5"] + P.T5_RUNOFF_RESPONSE_K * T5_local_runoff


# =====================================================================
# 6. Daily integration loop
# =====================================================================
v_pond = {nid: PONDS[nid]["v0"] for nid in POND_IDS}
v_tsf  = {nid: TSF[nid]["v0"]   for nid in TSF_IDS}

prev = {
    "T1_out":   WWTP["T1"]["base"],
    "T2_out":   WWTP["T2"]["base"],
    "R6_dec":   P.TSF_DECANT_K["R6"] * TSF["R6"]["v0"],
    "R7_dec":   P.TSF_DECANT_K["R7"] * TSF["R7"]["v0"],
    "T3_out":   0.0,
    "R7_to_R6": 0.0,
}

records = []

for i, dt in enumerate(all_dates):
    row = {"date": dt}

    # ---- Source-node runoff -----------------------------------------
    s_now = {sid: float(S_FLOW[sid][i]) for sid in SOURCES}
    row.update(s_now)

    # ---- Collection ponds  ------------------------------
    pond_in = {
        "R1": s_now.get("S4", 0.0),
        "R2": s_now.get("S5", 0.0) + s_now.get("S7", 0.0),
        "R3": s_now.get("S2", 0.0) + s_now.get("S6", 0.0),
    }
    r_pump, r_spill = {}, {}
    for nid in POND_IDS:
        q_pump, q_spill, v_new = pond_step(v_pond[nid], pond_in[nid],
                                           PONDS[nid])
        v_pond[nid]  = v_new
        r_pump[nid]  = q_pump
        r_spill[nid] = q_spill
        row[nid] = q_pump

    # ---- WWTP T1: pumps from R1/R2/R3 + process baseflow ------------
    t1_in = (r_pump["R1"] + r_pump["R2"] + r_pump["R3"]
             + WWTP["T1"]["base"])
    t1_out, t1_byp = wwtp_step(t1_in, prev["T1_out"], WWTP["T1"])
    row["T1"], row["T1_overflow"] = t1_out, t1_byp
    row["T1_overflow_flag"] = int(t1_byp > 0.0)

    # ---- WWTP T2: S1 direct + 0.32*T1 recycle + process baseflow ----
    t2_in = (s_now.get("S1", 0.0)
             + split[("T1", "T2")] * t1_out
             + WWTP["T2"]["base"])
    t2_out, t2_byp = wwtp_step(t2_in, prev["T2_out"], WWTP["T2"])
    row["T2"], row["T2_overflow"] = t2_out, t2_byp
    row["T2_overflow_flag"] = int(t2_byp > 0.0)

    # ---- T5 (new plant) discharges directly to R7 -------------------
    t5_out = float(T5_SERIES[i])
    row["T5"] = t5_out

    # ---- TSF R6 -----------------------------------------------------
    r6_process_in = (s_now.get("S3", 0.0)
                     + split[("T2", "R6")] * t2_out
                     + prev["R7_to_R6"])
    r6_state, _ = tsf_step(v_tsf["R6"], r6_process_in, i,
                           TSF["R6"], TSF_RAIN_ROUTED["R6"],
                           prev["R6_dec"])
    v_tsf["R6"] = r6_state["v"]
    row["R6"] = r6_state["decant"]

    # ---- TSF R7 -----------------------------------------------------
    r7_process_in = t5_out + prev["T3_out"]
    r7_state, _ = tsf_step(v_tsf["R7"], r7_process_in, i,
                           TSF["R7"], TSF_RAIN_ROUTED["R7"],
                           prev["R7_dec"])
    v_tsf["R7"] = r7_state["v"]
    row["R7"] = r7_state["decant"]

    # ---- T3 concentrator (1-day delay on inputs from R6/R7) ---------
    t3_in_today = (split[("R6", "T3")] * prev["R6_dec"]
                   + split[("R7", "T3")] * prev["R7_dec"])
    t3_out = min(t3_in_today, P.T3_MAX_THROUGHPUT)
    row["T3"] = t3_out

    # ---- T4 concentrator (1-day delay, no return) -------------------
    row["T4"] = split[("R6", "T4")] * prev["R6_dec"]

    # ---- Outfalls ---------------------------------------------------
    row["O1"] = (split[("T1", "O1")] * t1_out * (1.0 - P.WWTP_EFFLUENT_REUSE["T1"])
                 + P.WWTP_BYPASS_FRACTION * t1_byp)
    row["O2"] = (split[("T2", "O2")] * t2_out * (1.0 - P.WWTP_EFFLUENT_REUSE["T2"])
                 + P.WWTP_BYPASS_FRACTION * t2_byp)
    row["O3"] = (split[("R6", "O3")] * r6_state["decant"]
                 + split[("R7", "O3")] * r7_state["decant"]
                 + P.TSF_TOEDRAIN_TO_O3 * (r6_state["seep"]
                                           + r7_state["seep"]))

    prev["T1_out"]   = t1_out
    prev["T2_out"]   = t2_out
    prev["R6_dec"]   = r6_state["decant"]
    prev["R7_dec"]   = r7_state["decant"]
    prev["T3_out"]   = split[("T3", "R7")] * t3_out
    prev["R7_to_R6"] = split[("R7", "R6")] * r7_state["decant"]

    records.append(row)


# =====================================================================
# 7. Persist outputs
# =====================================================================
df = pd.DataFrame(records).set_index("date")

node_cols = ["S1", "S2", "S3", "S4", "S5", "S6", "S7",
             "R1", "R2", "R3", "R6", "R7",
             "T1", "T2", "T3", "T4", "T5",
             "O1", "O2", "O3"]
ovfl_cols = ["T1_overflow", "T1_overflow_flag",
             "T2_overflow", "T2_overflow_flag"]
for c in node_cols + ovfl_cols:
    if c not in df.columns:
        df[c] = 0.0

df = df[node_cols + ovfl_cols]
df.to_csv(OUT_FILE, encoding="utf-8-sig", date_format="%Y-%m-%d")

print(f"OK -> {OUT_FILE}")
print(f"   inputs: nodes={NODES_PATH.name}, "
      f"sub={SUB_PATH.name}, edges={EDGES_PATH.name}")
print(f"   period: {df.index.min().date()} .. {df.index.max().date()} "
      f"  rows={len(df)}")

print("\n=== Outfall summary (10^4 m^3/d) ===")
print(df[["O1", "O2", "O3"]]
      .describe().T[["mean", "std", "min", "max"]].round(3))
print("\n=== Outfall pairwise correlation ===")
print(df[["O1", "O2", "O3"]].corr().round(3))
print("\n=== SWAT inputs used ===")
print(", ".join(P.HYDRO_COLS))
