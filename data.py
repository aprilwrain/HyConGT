from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ops_data import (
    load_dexing_monthly_plan,
    load_jiama_monthly_plan,
    load_jiama_pool_storage,
)


SWAT_COLS = [
    "PRECIPmm",
    "SNOWMELTmm",
    "PETmm",
    "ETmm",
    "SWmm",
    "PERCmm",
    "SURQmm",
    "GW_Qmm",
    "WYLDmm",
    "LAT_Qmm",
]


@dataclass
class SiteBundle:
    site: str
    node_ids: list[str]
    dates: pd.DatetimeIndex
    X_dyn: np.ndarray
    X_static: np.ndarray
    feature_names: list[str]
    precip_mm: np.ndarray
    snowmelt_mm: np.ndarray
    pet_mm: np.ndarray
    area_m2: np.ndarray
    c0: np.ndarray
    vmax: np.ndarray
    asurf: np.ndarray
    qmine_base: np.ndarray
    qintake_base: np.ndarray
    q_recycle_base: np.ndarray
    qcap: np.ndarray
    is_s: np.ndarray
    is_r: np.ndarray
    is_t: np.ndarray
    is_o: np.ndarray
    is_surface_s: np.ndarray
    is_mine_s: np.ndarray
    is_intake_s: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_alpha: np.ndarray
    Y: np.ndarray # measured daily labels where available; missing entries are NaN
    observed_nodes: list[str]
    ops_scale: np.ndarray # (T, N) exogenous recycle intensity
    pool_drive: np.ndarray # (T, N) pond storage lag (0 if N/A)
    q_plan_recycle: np.ndarray # (T, N) monthly planned recycle m3/d per node
    q_plan_treat: np.ndarray # (T, N) monthly planned treat/throughput m3/d per node
    supervise_mode: str = "daily"


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ('utf-8-sig','gb18030'):
        try:
            return pd.read_csv(path,encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unsupported CSV encoding: {Path(path).name}')


def _yyyddd_to_date(value) -> pd.Timestamp:
    text=str(value).strip()
    if text.endswith('.0'): text=text[:-2]
    if len(text)!=7 or not text.isdigit(): raise ValueError('YYYYDDD requires year plus three-digit day')
    start=pd.Timestamp(year=int(text[:4]),month=1,day=1)
    day=int(text[4:])
    if not 1 <= day <= (366 if start.is_leap_year else 365): raise ValueError('Invalid ordinal day')
    return start+pd.Timedelta(days=day-1)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def default_splits(site: str) -> dict[str, str]:
    if site == "dexing":
        return {
            "train_start": "2021-01-01",
            "train_end": "2023-12-31",
            "valid_start": "2024-01-01",
            "valid_end": "2024-12-31",
            "test_start": "2025-01-01",
            "test_end": "2025-07-31",
        }
    if site == "jiama":
        return {
            "train_start": "2020-01-01",
            "train_end": "2024-08-31",
            "valid_start": "2024-09-01",
            "valid_end": "2024-12-31",
            "test_start": "2025-01-01",
            "test_end": "2025-08-31",
        }
    raise ValueError(site)


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row or pd.isna(row[col]):
        return default
    try:
        v = float(row[col])
    except Exception:
        return default
    miss = row.get(f"{col}_ismissing", None)
    if miss is not None and float(miss) == 1 and (v == 0 or pd.isna(row[col])):
        return default
    return v


def _classify_s(name: str, node_id: str, qmine: float) -> tuple[bool, bool, bool]:
    """Return (is_surface, is_mine, is_intake)."""
    n = name.lower()
    if "intake" in n or "取水" in n:
        return False, False, True
    if "underground" in n or "地下" in n or qmine > 0:
        return False, True, False
    # open-pit / waste rock / dump / default surface source
    return True, False, False


def _resolve_surface_area(row: pd.Series, site: str, is_surface_s: bool) -> float:
    """A1: only surface S nodes get catchment area for runoff."""
    if not is_surface_s:
        return 0.0
    if site == "dexing":
        for col in ("jiangyuhuishuimianji_m2", "shuimianmianji_m2"):
            if col in row and pd.notna(row[col]) and float(row[col]) > 0:
                miss = row.get(f"{col}_ismissing", 0)
                if float(miss) == 0:
                    return float(row[col])
        if pd.notna(row.get("area_km2", np.nan)) and float(row.get("area_km2_ismissing", 1)) == 0:
            return float(row["area_km2"]) * 1e6
        return 0.0
    # jiama: area_km2 * 1e6; invalid -> 0
    if pd.notna(row.get("area_km2", np.nan)) and float(row.get("area_km2", 0)) > 0:
        if float(row.get("area_km2_ismissing", 0)) == 0:
            return float(row["area_km2"]) * 1e6
    return 0.0


def load_site(site: str, data_root: Path) -> SiteBundle:
    site = site.lower()
    d = Path(data_root) / site
    if site == "dexing":
        nodes = _clean_columns(_read_csv(d / "dx_node.csv"))
        edge_path = d / "dx_edge_recalib.csv"
        if not edge_path.exists():
            edge_path = d / "dx_edge.csv"
        edges = _clean_columns(_read_csv(edge_path))
        sub = _clean_columns(_read_csv(d / "dx_sub.csv"))
    elif site == "jiama":
        nodes = _clean_columns(_read_csv(d / "jiama_node.csv"))
        edge_path = d / "jiama_edge_recalib.csv"
        if not edge_path.exists():
            edge_path = d / "jiama_edge.csv"
        edges = _clean_columns(_read_csv(edge_path))
        sub = _clean_columns(_read_csv(d / "jiama_sub.csv"))
    else:
        raise ValueError(site)

    if nodes['node_id'].isna().any() or nodes['node_id'].astype(str).duplicated().any():
        raise ValueError('Missing or duplicate node IDs')
    if not nodes['node_type'].astype(str).str.upper().isin(['S','R','T','O']).all():
        raise ValueError('Unknown SRTO node type')
    if edges.duplicated(['from_id','to_id']).any(): raise ValueError('Duplicate engineering edges')
    if not set(edges['from_id'].astype(str)).union(edges['to_id'].astype(str)).issubset(set(nodes['node_id'].astype(str))):
        raise ValueError('Engineering edge references unknown node')
    edge_values=pd.to_numeric(edges['weight_alpha'],errors='raise').to_numpy(dtype=float)
    if not np.isfinite(edge_values).all() or (edge_values<=0).any(): raise ValueError('Engineering weights must be finite and positive')
    nodes["node_id"] = nodes["node_id"].astype(str)
    node_ids = nodes["node_id"].tolist()
    n_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)

    sub["date"] = sub["YYYYDDD"].map(_yyyddd_to_date)
    dates = pd.DatetimeIndex(sorted(sub["date"].unique()))
    T = len(dates)
    date_to_t = {d: i for i, d in enumerate(dates)}
    if sub.duplicated(['date','SUB']).any(): raise ValueError('Duplicate SWAT date/subbasin rows')
    if len(dates)==0 or (len(dates)>1 and not np.all(np.diff(dates.values)==np.timedelta64(1,'D'))):
        raise ValueError('SWAT dates must have contiguous daily coverage')
    for col in SWAT_COLS:
        if col not in sub: raise ValueError(f'Missing SWAT field: {col}')
        values=pd.to_numeric(sub[col],errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values<0).any(): raise ValueError(f'Invalid SWAT values: {col}')
    if nodes['SUB'].isna().any() or not np.equal(nodes['SUB'], np.floor(nodes['SUB'])).all():
        raise ValueError('Each node needs an integer SWAT SUB identifier')
    for sub_id in nodes['SUB'].astype(int).unique():
        if len(sub.loc[sub['SUB']==sub_id])!=T: raise ValueError('Incomplete mapped SWAT subbasin coverage')
    training=default_splits(site)
    feature_train=np.asarray((dates>=training['train_start']) & (dates<=training['train_end']))
    if not feature_train.any(): raise ValueError('No training dates for input preprocessing')
    def train_median(values):
        selected=np.asarray(values)[feature_train]
        selected=selected[selected>0]
        return float(np.median(selected)) if len(selected) else 1.0
    sub_indexed = sub.set_index(["date", "SUB"])

    precip_mm = np.zeros((T, N), dtype=np.float32)
    snowmelt_mm = np.zeros((T, N), dtype=np.float32)
    pet_mm = np.zeros((T, N), dtype=np.float32)
    dyn_names = SWAT_COLS + ["month_sin", "month_cos", "doy_sin", "doy_cos"]
    X_dyn_list = []

    for t, dt in enumerate(dates):
        month, doy = dt.month, dt.dayofyear
        cal = [
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
            np.sin(2 * np.pi * doy / 366.0),
            np.cos(2 * np.pi * doy / 366.0),
        ]
        day_feats = np.zeros((N, len(dyn_names)), dtype=np.float32)
        for _, row in nodes.iterrows():
            ni = n_idx[str(row["node_id"])]
            sub_id = int(row["SUB"]) if pd.notna(row["SUB"]) else -1
            sw = None
            if sub_id >= 0 and (dt, sub_id) in sub_indexed.index:
                sw = sub_indexed.loc[(dt, sub_id)]
                if isinstance(sw, pd.DataFrame):
                    sw = sw.iloc[0]
            vals = []
            for c in SWAT_COLS:
                if sw is not None and c in sw.index and pd.notna(sw[c]):
                    vals.append(float(sw[c]))
                else:
                    vals.append(0.0)
            day_feats[ni, : len(SWAT_COLS)] = vals
            day_feats[ni, len(SWAT_COLS) :] = cal
            precip_mm[t, ni] = vals[SWAT_COLS.index("PRECIPmm")]
            snowmelt_mm[t, ni] = vals[SWAT_COLS.index("SNOWMELTmm")]
            pet_mm[t, ni] = vals[SWAT_COLS.index("PETmm")]
        X_dyn_list.append(day_feats)
    X_dyn = np.stack(X_dyn_list, axis=0)

    # Causal climate memory (precipitation only — not target-derived)
    precip_site = precip_mm.mean(axis=1, keepdims=True) # (T,1)
    roll_feats = []
    roll_names = []
    for w in (3, 7, 15, 30):
        cs = np.cumsum(precip_site, axis=0)
        pad = np.zeros((w, 1), dtype=np.float32)
        cs_pad = np.concatenate([pad, cs], axis=0)
        roll = ((cs_pad[w:] - cs_pad[:-w]) / float(w)).astype(np.float32)
        roll_feats.append(np.repeat(roll, N, axis=1)[..., None])
        roll_names.append(f"precip_roll{w}")
    X_roll = np.concatenate(roll_feats, axis=-1) # (T,N,4)
    X_dyn = np.concatenate([X_dyn, X_roll], axis=-1)
    dyn_names = dyn_names + roll_names

    # Monthly operational proxies, distinct from gauge-flow observations.
    aux = Path(data_root) / "aux"
    ops_feats = []
    ops_names = []
    if site == "dexing":
        plan = load_dexing_monthly_plan(aux, dates)
        ops = {
            "recycle_rate": plan["plan_rate"],
            "new_water_m3d": plan["plan_fresh_m3d"],
            "recycle_water_m3d": plan["plan_recycle_m3d"],
            "treat_m3d": plan["plan_treat_m3d"],
        }
    else:
        plan = load_jiama_monthly_plan(aux, dates)
        ops = {
            "recycle_rate": plan["plan_rate"],
            "recycle_vol_m3d": plan["plan_recycle_m3d"],
            "loss_m3d": plan.get("plan_loss_m3d", np.zeros(T, dtype=np.float32)),
            "treat_m3d": plan["plan_treat_m3d"],
            "new_water_m3d": plan["plan_fresh_m3d"],
        }
    # absolute + log + relative-to-median plan channels for GAT
    for k in ("recycle_rate", "treat_m3d", "recycle_water_m3d" if site == "dexing" else "recycle_vol_m3d"):
        key = k if k in ops else ("recycle_vol_m3d" if "recycle" in k else k)
        if key not in ops:
            continue
        raw = ops[key].astype(np.float32)
        ch = np.repeat(raw[:, None], N, axis=1)
        ops_feats.append(ch[..., None])
        ops_names.append(f"plan_{key}")
        if key != "recycle_rate":
            logc = np.log1p(np.maximum(raw, 0.0)).astype(np.float32)
            ops_feats.append(np.repeat(logc[:, None], N, axis=1)[..., None])
            ops_names.append(f"plan_log_{key}")
            med = train_median(raw)
            rel = (raw / max(med, 1.0)).astype(np.float32)
            ops_feats.append(np.repeat(rel[:, None], N, axis=1)[..., None])
            ops_names.append(f"plan_rel_{key}")
    # month cyclical encoding (plan calendar)
    month = np.array([d.month for d in dates], dtype=np.float32)
    mon_sin = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    mon_cos = np.cos(2 * np.pi * month / 12.0).astype(np.float32)
    ops_feats.append(np.repeat(mon_sin[:, None], N, axis=1)[..., None])
    ops_names.append("plan_month_sin")
    ops_feats.append(np.repeat(mon_cos[:, None], N, axis=1)[..., None])
    ops_names.append("plan_month_cos")
    if site == "jiama":
        pool = load_jiama_pool_storage(aux, dates)
        ops_feats.append(np.repeat(pool[:, None], N, axis=1)[..., None])
        ops_names.append("ops_pool_storage_lag1")
    if ops_feats:
        X_dyn = np.concatenate([X_dyn, np.concatenate(ops_feats, axis=-1)], axis=-1)
        dyn_names = dyn_names + ops_names

    # physics exogenous channels from monthly plan intensity
    rec_plan = ops.get("recycle_water_m3d", ops.get("recycle_vol_m3d", np.zeros(T, dtype=np.float32)))
    rec_plan = np.asarray(rec_plan, dtype=np.float32)
    med_rec = train_median(rec_plan)
    rate_ref = 0.92 if site == "dexing" else 0.78
    ops_scale = np.repeat(
        (ops["recycle_rate"] / rate_ref * (0.5 + 0.5 * rec_plan / max(med_rec, 1.0))).astype(np.float32)[:, None],
        N,
        axis=1,
    )
    ops_scale = np.clip(ops_scale, 0.2, 3.0).astype(np.float32)
    if site == "dexing":
        pool_drive = np.zeros((T, N), dtype=np.float32)
    else:
        pool = load_jiama_pool_storage(aux, dates)
        pool_drive = np.repeat(pool[:, None], N, axis=1).astype(np.float32)

    area_m2 = np.zeros(N, dtype=np.float32)
    c0 = np.zeros(N, dtype=np.float32)
    vmax = np.zeros(N, dtype=np.float32)
    asurf = np.zeros(N, dtype=np.float32)
    qmine_base = np.zeros(N, dtype=np.float32)
    qintake_base = np.zeros(N, dtype=np.float32)
    q_recycle_base = np.zeros(N, dtype=np.float32)
    qcap = np.full(N, 1e12, dtype=np.float32)
    is_s = np.zeros(N, dtype=np.float32)
    is_r = np.zeros(N, dtype=np.float32)
    is_t = np.zeros(N, dtype=np.float32)
    is_o = np.zeros(N, dtype=np.float32)
    is_surface_s = np.zeros(N, dtype=np.float32)
    is_mine_s = np.zeros(N, dtype=np.float32)
    is_intake_s = np.zeros(N, dtype=np.float32)

    # site mean recycle rate for nodes missing huiyonglv
    site_rr = float(np.mean(ops["recycle_rate"][feature_train]))
    # water intensity m3/t ore (order-of-magnitude prior; k_rec scales it)
    w0 = 1.8 if site == "dexing" else 0.45

    static_rows = []
    for _, row in nodes.iterrows():
        ni = n_idx[str(row["node_id"])]
        ntype = str(row["node_type"]).upper()
        name = str(row.get("name", ""))
        # jiama table uses m3/h; dexing may use m3/d
        qmine_h = max(_safe(row, "kuangkengyongshui_m3/h", 0.0), 0.0)
        qmine_d = max(_safe(row, "kuangkengyongshui_m3/d", 0.0), 0.0)
        qmine = qmine_d if qmine_d > 0 else qmine_h * 24.0
        surf, mine, intake = _classify_s(name, str(row["node_id"]), qmine)

        is_s[ni] = 1.0 if ntype == "S" else 0.0
        is_r[ni] = 1.0 if ntype == "R" else 0.0
        is_t[ni] = 1.0 if ntype == "T" else 0.0
        is_o[ni] = 1.0 if ntype == "O" else 0.0
        if ntype == "S":
            is_surface_s[ni] = 1.0 if surf else 0.0
            is_mine_s[ni] = 1.0 if mine else 0.0
            is_intake_s[ni] = 1.0 if intake else 0.0

        area_m2[ni] = _resolve_surface_area(row, site, bool(is_surface_s[ni]))
        c0[ni] = _safe(row, "jingliuxishu", 0.0) if is_surface_s[ni] else 0.0
        if is_surface_s[ni] and c0[ni] <= 0:
            c0[ni] = 0.5
        vmax[ni] = max(_safe(row, "youxiaokurong_m3", 0.0), 0.0) if ntype == "R" else 0.0
        asurf[ni] = max(_safe(row, "shuimianmianji_m2", 0.0), 0.0) if ntype == "R" else 0.0
        spill = max(_safe(row, "yihongkoubiaogao_m", 0.0), 0.0) if ntype == "R" else 0.0
        if vmax[ni] > 0:
            if asurf[ni] > 0:
                depth_proxy = 0.6
                if spill > 50: # absolute elev — use mild depth proxy only
                    depth_proxy = 0.8
                vmax[ni] = float(min(vmax[ni], max(asurf[ni] * depth_proxy, 3e4), 4.0e5))
            else:
                vmax[ni] = float(min(vmax[ni], 3.0e5))
        qmine_base[ni] = qmine if is_mine_s[ni] else 0.0
        qintake_base[ni] = 0.0
        treat = _safe(row, "chulinengli_m3/d", 0.0)
        if treat <= 0:
            # dexing may use chlinengli_t/d as capacity proxy (convert loosely via w0)
            treat_t = _safe(row, "chlinengli_t/d", 0.0)
            if treat_t > 0 and ntype == "T":
                treat = treat_t * w0
        if treat > 0 and ntype == "T":
            qcap[ni] = treat
        pump_h = _safe(row, "zuidabengpainengli_m3/h", 0.0)

        kaicai = max(_safe(row, "kaicainengli_t/d", 0.0), 0.0)
        # some tables store annual capacity under a daily column name
        if kaicai > 1.0e5:
            kaicai = kaicai / 365.0
        huiyong = _safe(row, "huiyonglv_%", np.nan)
        if not np.isfinite(huiyong) or huiyong <= 0:
            huiyong = site_rr * 100.0
        huiyong = float(np.clip(huiyong, 0.0, 99.5))
        # production recycle base on S with mining capacity; also allow R/T with huiyong as soft prior via static feats
        if ntype == "S" and kaicai > 0:
            q_recycle_base[ni] = float(kaicai * w0 * (huiyong / 100.0))
        elif ntype == "S" and is_mine_s[ni] and qmine > 0:
            q_recycle_base[ni] = float(qmine * 0.5 * (huiyong / 100.0))
        else:
            q_recycle_base[ni] = 0.0
        q_recycle_base[ni] = float(min(q_recycle_base[ni], 2.0e4))

        static_rows.append(
            [
                area_m2[ni],
                c0[ni],
                vmax[ni],
                asurf[ni],
                qmine_base[ni],
                qcap[ni] if qcap[ni] < 1e11 else 0.0,
                pump_h,
                kaicai,
                huiyong / 100.0,
                spill,
                q_recycle_base[ni],
                is_s[ni],
                is_r[ni],
                is_t[ni],
                is_o[ni],
                is_surface_s[ni],
                is_mine_s[ni],
                is_intake_s[ni],
                1.0 if area_m2[ni] == 0 and ntype == "S" and surf else 0.0,
                1.0 if vmax[ni] == 0 and ntype == "R" else 0.0,
                1.0 if treat == 0 and ntype == "T" else 0.0,
                np.log1p(area_m2[ni]),
                np.log1p(vmax[ni]),
                np.log1p(kaicai),
                np.log1p(q_recycle_base[ni]),
            ]
        )

    X_static = np.asarray(static_rows, dtype=np.float32)
    static_names = [
        "area_m2",
        "c0",
        "vmax",
        "asurf",
        "qmine_base",
        "qcap",
        "qpump_h",
        "kaicai",
        "huiyong",
        "spill_elev",
        "q_recycle_base",
        "S",
        "R",
        "T",
        "O",
        "surface_s",
        "mine_s",
        "intake_s",
        "area_missing",
        "vmax_missing",
        "qcap_missing",
        "log_area",
        "log_vmax",
        "log_kaicai",
        "log_qrec",
    ]

    # Distribute site planned recycle/treatment onto Source nodes.
    rec_series = ops.get("recycle_water_m3d", ops.get("recycle_vol_m3d", np.zeros(T)))
    rec_series = np.asarray(rec_series, dtype=np.float32)
    treat_series = np.asarray(ops.get("treat_m3d", np.zeros(T)), dtype=np.float32)
    if q_recycle_base.sum() > 0:
        shares = q_recycle_base / q_recycle_base.sum()
    else:
        shares = is_surface_s / max(float(is_surface_s.sum()), 1.0)
    # per-node planned recycle / treat (m3/d), capped so plant-wide totals don't explode one node
    q_plan_recycle = np.minimum(rec_series[:, None] * shares[None, :], 2.0e4).astype(np.float32)
    q_plan_treat = np.minimum(treat_series[:, None] * shares[None, :], 2.5e4).astype(np.float32)
    med = train_median(rec_series) if np.any(rec_series[feature_train] > 0) else 0.0
    if med > 0 and q_recycle_base.sum() > 0:
        q_recycle_base = np.maximum(
            q_recycle_base,
            (shares * min(med, 3.0e4)).astype(np.float32),
        )
        q_recycle_base = np.minimum(q_recycle_base, 2.5e4).astype(np.float32)

    srcs, dsts, alphas = [], [], []
    for _, er in edges.iterrows():
        s, t = str(er["from_id"]), str(er["to_id"])
        a = float(er.get("weight_alpha", 1.0) or 0.0)
        if s not in n_idx or t not in n_idx or a <= 0:
            continue
        srcs.append(n_idx[s])
        dsts.append(n_idx[t])
        alphas.append(a)
    edge_src = np.asarray(srcs, dtype=np.int64)
    edge_dst = np.asarray(dsts, dtype=np.int64)
    edge_alpha = np.asarray(alphas, dtype=np.float32)

    Y = read_daily_observations(d/'daily_observed.csv', dates, node_ids)
    observed_nodes = [node_ids[j] for j in range(N) if np.isfinite(Y[:,j]).any()]
    supervise_mode = 'daily'

    return SiteBundle(
        site=site,
        node_ids=node_ids,
        dates=dates,
        X_dyn=X_dyn,
        X_static=X_static,
        feature_names=dyn_names + [f"static:{n}" for n in static_names],
        precip_mm=precip_mm,
        snowmelt_mm=snowmelt_mm,
        pet_mm=pet_mm,
        area_m2=area_m2,
        c0=c0,
        vmax=vmax,
        asurf=asurf,
        qmine_base=qmine_base,
        qintake_base=qintake_base,
        q_recycle_base=q_recycle_base,
        qcap=qcap,
        is_s=is_s,
        is_r=is_r,
        is_t=is_t,
        is_o=is_o,
        is_surface_s=is_surface_s,
        is_mine_s=is_mine_s,
        is_intake_s=is_intake_s,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_alpha=edge_alpha,
        Y=Y,
        observed_nodes=observed_nodes,
        ops_scale=ops_scale,
        pool_drive=pool_drive,
        q_plan_recycle=q_plan_recycle,
        q_plan_treat=q_plan_treat,
        supervise_mode=supervise_mode,
    )


def read_daily_observations(path, dates, node_ids):
    """Read the daily-flow observation table into date/node order."""
    frame=_read_csv(path)
    required={'date','node_id','flow_m3_d'}
    if not required.issubset(frame): raise ValueError('daily_observed.csv needs date,node_id,flow_m3_d')
    frame['date']=pd.to_datetime(frame['date'],format='%Y-%m-%d',errors='raise')
    if frame['date'].isna().any() or frame['node_id'].isna().any(): raise ValueError('Missing observation keys')
    frame['node_id']=frame['node_id'].astype(str)
    if frame.duplicated(['date','node_id']).any(): raise ValueError('Duplicate daily observations')
    if not set(frame['node_id']).issubset(node_ids): raise ValueError('Daily observations reference unknown nodes')
    if not frame['date'].isin(dates).all(): raise ValueError('Daily observation outside forcing period')
    values=pd.to_numeric(frame['flow_m3_d'],errors='raise').to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)]<0).any(): raise ValueError('Invalid daily discharge')
    table=frame.pivot(index='date',columns='node_id',values='flow_m3_d')
    return table.reindex(index=dates,columns=node_ids).to_numpy(dtype=np.float32)
