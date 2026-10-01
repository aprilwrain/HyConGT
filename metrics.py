from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_pair(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape or np.isinf(y).any(): raise ValueError('Invalid paired arrays')
    m = np.isfinite(y)
    if not np.isfinite(p[m]).all(): raise ValueError('Nonfinite prediction at an observed date')
    return y[m], p[m]


def r2_score(y: np.ndarray, p: np.ndarray) -> float:
    """Pearson correlation squared (R²). Differs from NSE when there is bias."""
    y, p = _finite_pair(y, p)
    if len(y) < 2:
        return float("nan")
    sy, sp = float(np.std(y)), float(np.std(p))
    if sy <= 1e-12 or sp <= 1e-12:
        return float("nan")
    r = float(np.clip(np.mean((y-y.mean())*(p-p.mean()))/(sy*sp), -1, 1))
    if not np.isfinite(r):
        return float("nan")
    return float(r * r)


def nse_score(y: np.ndarray, p: np.ndarray) -> float:
    """Nash–Sutcliffe efficiency: 1 - SS_res / SS_tot (obs mean)."""
    y, p = _finite_pair(y, p)
    if len(y) < 2:
        return float("nan")
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse_score(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _finite_pair(y, p)
    if len(y) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae_score(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _finite_pair(y, p)
    if len(y) == 0:
        return float("nan")
    return float(np.mean(np.abs(y - p)))


def kge_score(y: np.ndarray, p: np.ndarray) -> float:
    y, p = _finite_pair(y, p)
    if len(y) < 2:
        return float("nan")
    sy, sp = float(np.std(y)), float(np.std(p))
    my, mp = float(np.mean(y)), float(np.mean(p))
    if sy <= 1e-12 or sp <= 1e-12 or abs(my) <= 1e-12:
        return float("nan")
    r = float(np.clip(np.mean((y-y.mean())*(p-p.mean()))/(sy*sp), -1, 1))
    alpha = sp / sy
    beta = mp / my
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def metrics_dict(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "R2": r2_score(y, p),
        "NSE": nse_score(y, p),
        "KGE": kge_score(y, p),
        "RMSE": rmse_score(y, p),
        "MAE": mae_score(y, p),
        "n": int(np.sum(np.isfinite(y) & np.isfinite(p))),
    }


def metrics_table(pred_df: pd.DataFrame, y_col: str = "observed_m3_d", p_col: str = "pred_m3_d") -> pd.DataFrame:
    rows = []
    obs = pred_df[np.isfinite(pred_df[y_col])].copy()
    for node_id, g in obs.groupby("node_id"):
        row = {"node_id": node_id}
        row.update(metrics_dict(g[y_col].to_numpy(), g[p_col].to_numpy()))
        rows.append(row)
    if not obs.empty:
        row = {"node_id": "ALL_OBSERVED"}
        row.update(metrics_dict(obs[y_col].to_numpy(), obs[p_col].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)
