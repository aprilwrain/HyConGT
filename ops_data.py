"""Monthly plan-like exogenous drivers (proxy from plant reports; not gauge-flow AR)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ('utf-8-sig','gb18030'):
        try:
            return pd.read_csv(path,encoding=encoding,dtype=str)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Unsupported CSV encoding: {Path(path).name}')


def _ffill_series(a: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Causal fill only (no bfill — avoids leaking later observations into leading gaps)."""
    s = pd.Series(a).replace(0, np.nan)
    return s.ffill().fillna(fill).to_numpy(dtype=np.float32)


def _nonnegative(value):
    result = float(pd.to_numeric(value, errors='raise'))
    if not np.isfinite(result) or result < 0:
        raise ValueError('Operating values must be finite and nonnegative')
    return result


def load_dexing_monthly_plan(aux: Path, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Build daily series of plant water / recycle 'plans' from yearly report.

    Uses annual totals -> mean daily, modulated by a mild seasonal curve so each
    month has a distinct plan level (proxy for monthly planned throughput).
    """
    T = len(dates)
    rate = np.full(T, 0.92, dtype=np.float32)
    treat = np.zeros(T, dtype=np.float32) # proxy planned plant water m3/d
    recycle = np.zeros(T, dtype=np.float32)
    fresh = np.zeros(T, dtype=np.float32)
    xlsx = aux / "dx_recycle_2021_2023.xlsx"
    year_stats: dict[int, tuple[float, float, float, float]] = {}
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name=0)
        try:
            for year, col_acc in [(2021, 3), (2022, 5), (2023, 7)]:
                if col_acc >= df.shape[1]:
                    continue
                total = _nonnegative(df.iloc[0, col_acc])
                fr = _nonnegative(df.iloc[1, col_acc])
                rec = _nonnegative(df.iloc[2, col_acc])
                rr = _nonnegative(df.iloc[12, col_acc]) / 100.0
                if rr > 1: raise ValueError('Recycle percentage exceeds 100')
                year_stats[year] = (total / 365.0, fr / 365.0, rec / 365.0, rr)
        except (ValueError, TypeError, IndexError) as exc:
            raise ValueError("Invalid private annual operating table") from exc
    if not year_stats:
        raise ValueError("A valid private dx_recycle_2021_2023.xlsx is required; no fabricated operating fallback is used")

    doy = np.array([d.dayofyear for d in dates], dtype=np.float32)
    # seasonal multiplier ~0.85–1.15 (wet season higher plant load proxy)
    season = (1.0 + 0.15 * np.sin(2 * np.pi * (doy - 60) / 365.0)).astype(np.float32)
    for t, d in enumerate(dates):
        previous = [y for y in year_stats if y <= int(d.year)]
        if not previous: raise ValueError('No current or earlier annual operating report')
        tot, fr, rec, rr = year_stats[max(previous)]
        rate[t] = rr
        treat[t] = tot * season[t]
        fresh[t] = fr * season[t]
        recycle[t] = rec * season[t]
    # cap absolute injections used in physics (plant-wide totals are huge)
    treat_cap = np.minimum(treat, 1.2e5).astype(np.float32)
    recycle_cap = np.minimum(recycle, 1.0e5).astype(np.float32)
    return {
        "plan_rate": rate,
        "plan_treat_m3d": treat_cap,
        "plan_recycle_m3d": recycle_cap,
        "plan_fresh_m3d": np.minimum(fresh, 3.0e4).astype(np.float32),
        "plan_treat_raw": treat.astype(np.float32),
        "plan_recycle_raw": recycle.astype(np.float32),
    }


def load_jiama_monthly_plan(aux: Path, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Monthly T2 volumes -> constant-within-month planned recycle / loss."""
    T = len(dates)
    out = {
        "plan_rate": np.full(T, 0.78, dtype=np.float32),
        "plan_treat_m3d": np.zeros(T, dtype=np.float32),
        "plan_recycle_m3d": np.zeros(T, dtype=np.float32),
        "plan_fresh_m3d": np.zeros(T, dtype=np.float32),
        "plan_loss_m3d": np.zeros(T, dtype=np.float32),
    }
    path = aux / "jm_T2_monthly.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv(path)
    cols = list(df.columns)
    if len(cols) < 3: raise ValueError('Monthly operations need date, volume and rate columns')
    date_col, rate_col = cols[0], cols[-1]
    vol_cols = [c for c in cols[1:-1] if "失" not in str(c)]
    loss_cols = [c for c in cols[1:-1] if "失" in str(c)]
    seen = set()
    for _, row in df.iterrows():
        raw = str(row[date_col]).strip().replace("/", ".")
        parts = raw.split(".")
        if len(parts) != 2:
            raise ValueError('Monthly operating date must be YYYY.MM')
        y, m = int(parts[0]), int(parts[1])
        if (y, m) in seen: raise ValueError('Duplicate operating month')
        seen.add((y, m))
        rate = _nonnegative(str(row[rate_col]).replace('%', '').strip()) / 100.0
        if rate > 1: raise ValueError('Recycle percentage exceeds 100')
        vol = sum(_nonnegative(row[c]) for c in vol_cols)
        loss = sum(_nonnegative(row[c]) for c in loss_cols)
        days = pd.Period(f"{y}-{m:02d}").days_in_month
        mask = (dates.year == y) & (dates.month == m)
        out["plan_rate"][mask] = rate
        out["plan_recycle_m3d"][mask] = vol / days
        out["plan_loss_m3d"][mask] = loss / days
        out["plan_treat_m3d"][mask] = (vol + loss) / days
    for k, fill in (
        ("plan_rate", 0.78),
        ("plan_treat_m3d", 0.0),
        ("plan_recycle_m3d", 0.0),
        ("plan_loss_m3d", 0.0),
    ):
        out[k] = _ffill_series(out[k], fill)
    out["plan_fresh_m3d"] = (out["plan_treat_m3d"] * (1.0 - out["plan_rate"])).astype(np.float32)
    # physics caps
    out["plan_treat_m3d"] = np.minimum(out["plan_treat_m3d"], 8.0e4).astype(np.float32)
    out["plan_recycle_m3d"] = np.minimum(out["plan_recycle_m3d"], 8.0e4).astype(np.float32)
    return out


def load_jiama_pool_storage(aux: Path, dates: pd.DatetimeIndex) -> np.ndarray:
    """回水池蓄量 lag1 — NOT O1/T5 gauge flows."""
    T = len(dates)
    pool = np.full(T, np.nan, dtype=np.float32)
    path = aux / "jm_T5_daily.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_csv(path)
    cols = list(df.columns)
    if len(cols) < 2: raise ValueError('Pool table needs date and storage columns')
    date_col, pool_col = cols[0], cols[1]
    date_to_i = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates)}
    seen = set()
    for _, row in df.iterrows():
        raw = str(row[date_col]).strip().replace("/", ".")
        parts = raw.split(".")
        if len(parts) != 3:
            raise ValueError('Pool date must be YYYY.MM.DD')
        dt = pd.Timestamp(int(parts[0]), int(parts[1]), int(parts[2])).normalize()
        if dt in seen: raise ValueError('Duplicate pool-storage date')
        seen.add(dt)
        if dt not in date_to_i:
            continue
        v = pd.to_numeric(row[pool_col], errors="raise")
        if pd.notna(v):
            pool[date_to_i[dt]] = _nonnegative(v)
    # Causal fill only: no bfill (bfill would leak future first observation into leading NaNs).
    filled = pd.Series(pool).ffill().fillna(0.0).to_numpy(dtype=np.float32)
    lag = np.zeros_like(filled)
    lag[1:] = filled[:-1]
    return lag


