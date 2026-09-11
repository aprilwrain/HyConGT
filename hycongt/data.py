from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_SPLITS: dict[str, dict[str, str]] = {
    "dexing": {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "valid_start": "2024-01-01",
        "valid_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-07-31",
    },
    "jiama": {
        "train_start": "2020-01-01",
        "train_end": "2024-08-31",
        "valid_start": "2024-09-01",
        "valid_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-08-31",
    },
}


@dataclass(frozen=True)
class SiteBundle:
    """Prepared, non-public model inputs for one engineered mine-water network.

    Raw mine records are intentionally outside the public repository. This structure
    is the boundary between confidential preprocessing and the public HyConGT model.
    """

    site: str
    node_ids: list[str]
    dates: pd.DatetimeIndex
    x_dynamic: np.ndarray
    x_static: np.ndarray
    dynamic_feature_names: list[str]
    static_feature_names: list[str]

    precip_mm: np.ndarray
    snowmelt_mm: np.ndarray
    pet_mm: np.ndarray
    gw_mm: np.ndarray
    ops_scale: np.ndarray
    pool_drive: np.ndarray
    q_plan_recycle: np.ndarray
    q_plan_treat: np.ndarray

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

    obs_daily: np.ndarray
    obs_monthly_total: np.ndarray
    supervision_mode: str

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def validate(self) -> None:
        t, n = self.n_days, self.n_nodes
        if self.site not in PAPER_SPLITS:
            raise ValueError(f"Unsupported site {self.site!r}. Expected one of {sorted(PAPER_SPLITS)}.")
        if self.supervision_mode not in {"daily", "monthly"}:
            raise ValueError("supervision_mode must be 'daily' or 'monthly'.")
        expected_mode = "monthly" if self.site == "dexing" else "daily"
        if self.supervision_mode != expected_mode:
            raise ValueError(
                f"{self.site} must use supervision_mode={expected_mode!r} to match the manuscript."
            )
        if not self.dates.is_monotonic_increasing or self.dates.has_duplicates:
            raise ValueError("dates must be strictly increasing and unique.")
        if len(self.dates) > 1:
            step_days = np.diff(self.dates.values).astype("timedelta64[D]").astype(int)
            if not np.all(step_days == 1):
                raise ValueError("dates must form a continuous daily sequence.")
        if self.x_dynamic.ndim != 3 or self.x_dynamic.shape[:2] != (t, n):
            raise ValueError(f"x_dynamic must have shape (T,N,F); got {self.x_dynamic.shape}.")
        if self.x_static.ndim != 2 or self.x_static.shape[0] != n:
            raise ValueError(f"x_static must have shape (N,F); got {self.x_static.shape}.")
        if len(self.dynamic_feature_names) != self.x_dynamic.shape[-1]:
            raise ValueError("dynamic_feature_names length does not match x_dynamic.")
        if len(self.static_feature_names) != self.x_static.shape[-1]:
            raise ValueError("static_feature_names length does not match x_static.")
        if not np.isfinite(self.x_dynamic).all():
            raise ValueError("x_dynamic must be finite after confidential preprocessing.")
        if not np.isfinite(self.x_static).all():
            raise ValueError("x_static must be finite after confidential preprocessing.")

        tn_arrays = {
            "precip_mm": self.precip_mm,
            "snowmelt_mm": self.snowmelt_mm,
            "pet_mm": self.pet_mm,
            "gw_mm": self.gw_mm,
            "ops_scale": self.ops_scale,
            "pool_drive": self.pool_drive,
            "q_plan_recycle": self.q_plan_recycle,
            "q_plan_treat": self.q_plan_treat,
            "obs_daily": self.obs_daily,
            "obs_monthly_total": self.obs_monthly_total,
        }
        for name, arr in tn_arrays.items():
            if arr.shape != (t, n):
                raise ValueError(f"{name} must have shape {(t, n)}; got {arr.shape}.")
            if name not in {"obs_daily", "obs_monthly_total"} and not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite after confidential preprocessing.")

        n_arrays = {
            "area_m2": self.area_m2,
            "c0": self.c0,
            "vmax": self.vmax,
            "asurf": self.asurf,
            "qmine_base": self.qmine_base,
            "qintake_base": self.qintake_base,
            "q_recycle_base": self.q_recycle_base,
            "qcap": self.qcap,
            "is_s": self.is_s,
            "is_r": self.is_r,
            "is_t": self.is_t,
            "is_o": self.is_o,
            "is_surface_s": self.is_surface_s,
            "is_mine_s": self.is_mine_s,
            "is_intake_s": self.is_intake_s,
        }
        for name, arr in n_arrays.items():
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape {(n,)}; got {arr.shape}.")

        if not (self.edge_src.ndim == self.edge_dst.ndim == self.edge_alpha.ndim == 1):
            raise ValueError("edge_src, edge_dst, and edge_alpha must be 1D arrays.")
        if not (len(self.edge_src) == len(self.edge_dst) == len(self.edge_alpha)):
            raise ValueError("edge_src, edge_dst, and edge_alpha must have the same length.")
        if len(self.edge_src) == 0:
            raise ValueError("At least one engineering edge is required.")
        if np.any(self.edge_src < 0) or np.any(self.edge_src >= n):
            raise ValueError("edge_src contains an invalid node index.")
        if np.any(self.edge_dst < 0) or np.any(self.edge_dst >= n):
            raise ValueError("edge_dst contains an invalid node index.")
        if np.any(~np.isfinite(self.edge_alpha)) or np.any(self.edge_alpha <= 0):
            raise ValueError("All baseline engineering edge weights must be finite and positive.")

        node_type_sum = self.is_s + self.is_r + self.is_t + self.is_o
        if not np.allclose(node_type_sum, 1.0, atol=1e-5):
            raise ValueError("Each node must belong to exactly one SRTO functional class.")
        source_subtype_sum = self.is_surface_s + self.is_mine_s + self.is_intake_s
        if np.any((self.is_s > 0.5) & (~np.isclose(source_subtype_sum, 1.0, atol=1e-5))):
            raise ValueError("Each Source node must have exactly one Source subtype.")
        if np.any((self.is_s < 0.5) & (source_subtype_sum > 1e-5)):
            raise ValueError("Non-Source nodes cannot carry a Source subtype.")
        if np.any(self.vmax < 0) or np.any(self.qcap < 0):
            raise ValueError("vmax and qcap must be non-negative.")


def _string_list(a: np.ndarray) -> list[str]:
    return [str(x) for x in np.asarray(a).tolist()]


def _scalar_string(a: np.ndarray) -> str:
    return str(np.asarray(a).reshape(-1)[0])


def load_prepared_site(path: str | Path) -> SiteBundle:
    """Load a confidentially prepared site bundle from a single NPZ file."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        required = {
            "site",
            "node_ids",
            "dates",
            "x_dynamic",
            "x_static",
            "dynamic_feature_names",
            "static_feature_names",
            "precip_mm",
            "snowmelt_mm",
            "pet_mm",
            "gw_mm",
            "ops_scale",
            "pool_drive",
            "q_plan_recycle",
            "q_plan_treat",
            "area_m2",
            "c0",
            "vmax",
            "asurf",
            "qmine_base",
            "qintake_base",
            "q_recycle_base",
            "qcap",
            "is_s",
            "is_r",
            "is_t",
            "is_o",
            "is_surface_s",
            "is_mine_s",
            "is_intake_s",
            "edge_src",
            "edge_dst",
            "edge_alpha",
            "obs_daily",
            "obs_monthly_total",
            "supervision_mode",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise KeyError(f"Prepared NPZ is missing required arrays {missing}.")

        bundle = SiteBundle(
            site=_scalar_string(z["site"]).lower(),
            node_ids=_string_list(z["node_ids"]),
            dates=pd.DatetimeIndex(pd.to_datetime(_string_list(z["dates"]))),
            x_dynamic=np.asarray(z["x_dynamic"], dtype=np.float32),
            x_static=np.asarray(z["x_static"], dtype=np.float32),
            dynamic_feature_names=_string_list(z["dynamic_feature_names"]),
            static_feature_names=_string_list(z["static_feature_names"]),
            precip_mm=np.asarray(z["precip_mm"], dtype=np.float32),
            snowmelt_mm=np.asarray(z["snowmelt_mm"], dtype=np.float32),
            pet_mm=np.asarray(z["pet_mm"], dtype=np.float32),
            gw_mm=np.asarray(z["gw_mm"], dtype=np.float32),
            ops_scale=np.asarray(z["ops_scale"], dtype=np.float32),
            pool_drive=np.asarray(z["pool_drive"], dtype=np.float32),
            q_plan_recycle=np.asarray(z["q_plan_recycle"], dtype=np.float32),
            q_plan_treat=np.asarray(z["q_plan_treat"], dtype=np.float32),
            area_m2=np.asarray(z["area_m2"], dtype=np.float32),
            c0=np.asarray(z["c0"], dtype=np.float32),
            vmax=np.asarray(z["vmax"], dtype=np.float32),
            asurf=np.asarray(z["asurf"], dtype=np.float32),
            qmine_base=np.asarray(z["qmine_base"], dtype=np.float32),
            qintake_base=np.asarray(z["qintake_base"], dtype=np.float32),
            q_recycle_base=np.asarray(z["q_recycle_base"], dtype=np.float32),
            qcap=np.asarray(z["qcap"], dtype=np.float32),
            is_s=np.asarray(z["is_s"], dtype=np.float32),
            is_r=np.asarray(z["is_r"], dtype=np.float32),
            is_t=np.asarray(z["is_t"], dtype=np.float32),
            is_o=np.asarray(z["is_o"], dtype=np.float32),
            is_surface_s=np.asarray(z["is_surface_s"], dtype=np.float32),
            is_mine_s=np.asarray(z["is_mine_s"], dtype=np.float32),
            is_intake_s=np.asarray(z["is_intake_s"], dtype=np.float32),
            edge_src=np.asarray(z["edge_src"], dtype=np.int64),
            edge_dst=np.asarray(z["edge_dst"], dtype=np.int64),
            edge_alpha=np.asarray(z["edge_alpha"], dtype=np.float32),
            obs_daily=np.asarray(z["obs_daily"], dtype=np.float32),
            obs_monthly_total=np.asarray(z["obs_monthly_total"], dtype=np.float32),
            supervision_mode=_scalar_string(z["supervision_mode"]).lower(),
        )
    bundle.validate()
    return bundle


def split_masks(bundle: SiteBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = PAPER_SPLITS[bundle.site]
    dates = bundle.dates
    train = np.asarray((dates >= split["train_start"]) & (dates <= split["train_end"]))
    valid = np.asarray((dates >= split["valid_start"]) & (dates <= split["valid_end"]))
    test = np.asarray((dates >= split["test_start"]) & (dates <= split["test_end"]))
    if not train.any() or not valid.any() or not test.any():
        raise ValueError("Prepared dates do not cover all paper train, validation, and test periods.")
    return train, valid, test


def standardize_dynamic(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x[train_mask].mean(axis=(0, 1), keepdims=True)
    sigma = x[train_mask].std(axis=(0, 1), keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return ((x - mu) / sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def standardize_static(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return ((x - mu) / sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)
