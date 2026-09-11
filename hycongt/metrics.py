from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_pair(observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(predicted)
    return observed[mask], predicted[mask]


def r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Squared Pearson correlation, matching Eq. S43."""

    y, p = _finite_pair(observed, predicted)
    if len(y) < 2 or np.std(y) <= 1e-12 or np.std(p) <= 1e-12:
        return float("nan")
    r = float(np.corrcoef(y, p)[0, 1])
    return float(r * r) if np.isfinite(r) else float("nan")


def nse_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    y, p = _finite_pair(observed, predicted)
    if len(y) < 2:
        return float("nan")
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / denominator


def kge_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    y, p = _finite_pair(observed, predicted)
    if len(y) < 2:
        return float("nan")
    sy, sp = float(np.std(y)), float(np.std(p))
    my, mp = float(np.mean(y)), float(np.mean(p))
    if sy <= 1e-12 or sp <= 1e-12 or abs(my) <= 1e-12:
        return float("nan")
    r = float(np.corrcoef(y, p)[0, 1])
    alpha = sp / sy
    beta = mp / my
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def rmse_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    y, p = _finite_pair(observed, predicted)
    return float(np.sqrt(np.mean((y - p) ** 2))) if len(y) else float("nan")


def mae_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    y, p = _finite_pair(observed, predicted)
    return float(np.mean(np.abs(y - p))) if len(y) else float("nan")


def metrics_dict(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    y, p = _finite_pair(observed, predicted)
    return {
        "R2": r2_score(y, p),
        "NSE": nse_score(y, p),
        "KGE": kge_score(y, p),
        "RMSE": rmse_score(y, p),
        "MAE": mae_score(y, p),
        "n": int(len(y)),
    }


def metrics_by_node(
    observed: np.ndarray,
    predicted: np.ndarray,
    node_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    pooled_y: list[np.ndarray] = []
    pooled_p: list[np.ndarray] = []
    for j, node_id in enumerate(node_ids):
        y, p = _finite_pair(observed[:, j], predicted[:, j])
        if len(y) == 0:
            continue
        row: dict[str, float | int | str] = {"node_id": node_id}
        row.update(metrics_dict(y, p))
        rows.append(row)
        pooled_y.append(y)
        pooled_p.append(p)
    if pooled_y:
        row = {"node_id": "ALL_OBSERVED"}
        row.update(metrics_dict(np.concatenate(pooled_y), np.concatenate(pooled_p)))
        rows.append(row)
    return pd.DataFrame(rows)
