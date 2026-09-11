from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import SiteBundle
from .metrics import metrics_by_node, nse_score
from .model import HyConGT


SLOW_PARAMETER_KEYS = (
    "k_mine",
    "k_intake",
    "k_rec",
    "r",
    "p",
    "eta",
    "q_op_latent",
)


class CompositeHuberLoss(nn.Module):
    """Huber loss on log flow plus a half-weighted scaled-flow Huber term."""

    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction="none")

    def forward(self, predicted: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() < 1:
            return predicted.new_zeros(())
        log_term = self.huber(
            torch.log1p(predicted.clamp_min(0.0)),
            torch.log1p(observed.clamp_min(0.0)),
        )
        scale = observed.abs().clamp_min(50.0)
        scaled_term = self.huber(predicted / scale, observed / scale)
        element = log_term + 0.5 * scaled_term
        active = mask > 0
        return element[active].mean()


@dataclass
class RolloutState:
    storage: torch.Tensor
    gru_hidden: torch.Tensor
    previous_q: torch.Tensor
    slow_params: dict[str, torch.Tensor] | None


def make_static_tensors(bundle: SiteBundle, x_static_std: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    names = (
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
    )
    out = {
        name: torch.as_tensor(getattr(bundle, name), dtype=torch.float32, device=device)
        for name in names
    }
    out["x_static"] = torch.as_tensor(x_static_std, dtype=torch.float32, device=device)
    return out


def make_physics_dynamic(bundle: SiteBundle) -> np.ndarray:
    return np.stack(
        [
            bundle.precip_mm,
            bundle.snowmelt_mm,
            bundle.pet_mm,
            bundle.gw_mm,
            bundle.ops_scale,
            bundle.pool_drive,
            bundle.q_plan_recycle,
            bundle.q_plan_treat,
        ],
        axis=-1,
    ).astype(np.float32)


def trailing_mean_targets(raw: np.ndarray, window: int = 7) -> np.ndarray:
    out = np.full_like(raw, np.nan, dtype=np.float32)
    for j in range(raw.shape[1]):
        out[:, j] = (
            pd.Series(raw[:, j], dtype=float)
            .rolling(window=window, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float32)
        )
    return out


def flow_log_scale(bundle: SiteBundle, train_mask: np.ndarray) -> float:
    if bundle.supervision_mode == "monthly":
        values = bundle.obs_monthly_total[train_mask]
        values = values[np.isfinite(values)] / 30.0
    else:
        values = bundle.obs_daily[train_mask]
        values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.log1p(np.nanmean(np.maximum(values, 0.0)))), 1.0)


def _qhat_feature(previous_q: torch.Tensor, q_scale: float) -> torch.Tensor:
    return (torch.log1p(previous_q.clamp_min(0.0)) / q_scale).unsqueeze(-1)


def _apply_temporal_inertia(
    current: dict[str, torch.Tensor],
    previous: dict[str, torch.Tensor] | None,
    rho: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    if rho <= 0.0:
        return current, None
    effective = dict(current)
    next_state: dict[str, torch.Tensor] = {}
    for key in SLOW_PARAMETER_KEYS:
        value = current[key]
        if previous is not None:
            value = rho * previous[key] + (1.0 - rho) * value
        effective[key] = value
        next_state[key] = value.detach()
    return effective, next_state


def initialize_state(model: HyConGT, static: dict[str, torch.Tensor], device: torch.device) -> RolloutState:
    return RolloutState(
        storage=model.initial_storage(static["vmax"], batch_size=1).to(device),
        gru_hidden=torch.zeros(1, model.n_nodes, model.gru.hidden_size, device=device),
        previous_q=torch.zeros(model.n_nodes, device=device),
        slow_params=None,
    )


def step_day(
    model: HyConGT,
    x_dynamic_t: torch.Tensor,
    physics_t: torch.Tensor,
    static: dict[str, torch.Tensor],
    edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    state: RolloutState,
    q_scale: float,
    slow_rho: float,
) -> tuple[torch.Tensor, RolloutState]:
    edge_src, edge_dst, edge_alpha = edges
    x_dynamic_t = torch.cat([x_dynamic_t, _qhat_feature(state.previous_q, q_scale)], dim=-1)
    node_input = torch.cat([x_dynamic_t, static["x_static"]], dim=-1).unsqueeze(0)
    spatial = model.encode_step(node_input)
    temporal, gru_hidden = model.gru(spatial.squeeze(0).unsqueeze(1), state.gru_hidden)
    hidden_nodes = temporal.squeeze(1).unsqueeze(0)
    current = model.params_from_hidden(hidden_nodes, edge_src, edge_dst)
    effective, slow_params = _apply_temporal_inertia(current, state.slow_params, slow_rho)

    q, storage = model.balance(
        precip_mm=physics_t[:, 0].unsqueeze(0),
        snowmelt_mm=physics_t[:, 1].unsqueeze(0),
        pet_mm=physics_t[:, 2].unsqueeze(0),
        gw_mm=physics_t[:, 3].unsqueeze(0),
        storage=state.storage,
        k_c=effective["k_c"],
        k_snow=effective["k_snow"],
        k_gw=effective["k_gw"],
        k_mine=effective["k_mine"],
        k_intake=effective["k_intake"],
        k_rec=effective["k_rec"],
        k_e=effective["k_e"],
        r=effective["r"],
        p=effective["p"],
        eta=effective["eta"],
        q_intake_latent=effective["q_intake_latent"],
        q_op_latent=effective["q_op_latent"],
        delta_alpha=effective["delta_alpha"],
        ops_scale=physics_t[:, 4].unsqueeze(0),
        pool_drive=physics_t[:, 5].unsqueeze(0),
        q_plan_recycle=physics_t[:, 6].unsqueeze(0),
        q_plan_treat=physics_t[:, 7].unsqueeze(0),
        area_m2=static["area_m2"],
        c0=static["c0"],
        vmax=static["vmax"],
        asurf=static["asurf"],
        qmine_base=static["qmine_base"],
        qintake_base=static["qintake_base"],
        q_recycle_base=static["q_recycle_base"],
        qcap=static["qcap"],
        is_s=static["is_s"],
        is_r=static["is_r"],
        is_t=static["is_t"],
        is_o=static["is_o"],
        is_surface_s=static["is_surface_s"],
        is_mine_s=static["is_mine_s"],
        is_intake_s=static["is_intake_s"],
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_alpha=edge_alpha,
    )
    q = q.squeeze(0)
    next_state = RolloutState(
        storage=storage,
        gru_hidden=gru_hidden,
        previous_q=q.detach(),
        slow_params=slow_params,
    )
    return q, next_state


def _fixed_chunks(length: int, chunk_days: int) -> list[tuple[int, int]]:
    return [(start, min(length, start + chunk_days)) for start in range(0, length, chunk_days)]


def _calendar_month_chunks(dates: pd.DatetimeIndex, end_exclusive: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    start = 0
    while start < end_exclusive:
        year, month = dates[start].year, dates[start].month
        end = start + 1
        while end < end_exclusive and dates[end].year == year and dates[end].month == month:
            end += 1
        chunks.append((start, end))
        start = end
    return chunks


def train_one_epoch(
    model: HyConGT,
    optimizer: torch.optim.Optimizer,
    criterion: CompositeHuberLoss,
    x_dynamic: np.ndarray,
    physics: np.ndarray,
    static: dict[str, torch.Tensor],
    edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bundle: SiteBundle,
    train_mask: np.ndarray,
    train_targets: np.ndarray,
    q_scale: float,
    slow_rho: float,
    chunk_days: int,
    lambda_month: float,
    lambda_day: float,
    device: torch.device,
) -> float:
    model.train()
    end_exclusive = int(np.where(train_mask)[0][-1]) + 1
    chunks = (
        _calendar_month_chunks(bundle.dates, end_exclusive)
        if bundle.supervision_mode == "monthly"
        else _fixed_chunks(end_exclusive, chunk_days)
    )
    state = initialize_state(model, static, device)
    losses: list[float] = []

    for start, end in chunks:
        if start > 0:
            state.storage = state.storage.detach()
            state.gru_hidden = state.gru_hidden.detach()
            state.previous_q = state.previous_q.detach()
            if state.slow_params is not None:
                state.slow_params = {k: v.detach() for k, v in state.slow_params.items()}

        optimizer.zero_grad(set_to_none=True)
        chunk_predictions: list[torch.Tensor] = []
        chunk_loss: torch.Tensor | None = None

        for t in range(start, end):
            x_t = torch.as_tensor(x_dynamic[t], dtype=torch.float32, device=device)
            phys_t = torch.as_tensor(physics[t], dtype=torch.float32, device=device)
            q, state = step_day(model, x_t, phys_t, static, edges, state, q_scale, slow_rho)
            chunk_predictions.append(q)

            if train_mask[t]:
                observed = train_targets[t]
                mask = np.isfinite(observed).astype(np.float32)
                if mask.sum() > 0:
                    y = torch.as_tensor(np.nan_to_num(observed, nan=0.0), dtype=torch.float32, device=device)
                    m = torch.as_tensor(mask, dtype=torch.float32, device=device)
                    loss_t = lambda_day * criterion(q, y, m)
                    chunk_loss = loss_t if chunk_loss is None else chunk_loss + loss_t

        if bundle.supervision_mode == "monthly" and np.all(train_mask[start:end]):
            monthly_observed = bundle.obs_monthly_total[end - 1]
            monthly_mask = np.isfinite(monthly_observed).astype(np.float32)
            if monthly_mask.sum() > 0:
                q_month = torch.stack(chunk_predictions, dim=0).sum(dim=0)
                y_month = torch.as_tensor(
                    np.nan_to_num(monthly_observed, nan=0.0), dtype=torch.float32, device=device
                )
                m_month = torch.as_tensor(monthly_mask, dtype=torch.float32, device=device)
                loss_month = lambda_month * criterion(q_month, y_month, m_month)
                chunk_loss = loss_month if chunk_loss is None else chunk_loss + loss_month

        if chunk_loss is not None:
            chunk_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(chunk_loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def rollout(
    model: HyConGT,
    x_dynamic: np.ndarray,
    physics: np.ndarray,
    static: dict[str, torch.Tensor],
    edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    q_scale: float,
    slow_rho: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    state = initialize_state(model, static, device)
    predictions: list[np.ndarray] = []
    for t in range(x_dynamic.shape[0]):
        x_t = torch.as_tensor(x_dynamic[t], dtype=torch.float32, device=device)
        phys_t = torch.as_tensor(physics[t], dtype=torch.float32, device=device)
        q, state = step_day(model, x_t, phys_t, static, edges, state, q_scale, slow_rho)
        predictions.append(q.cpu().numpy())
    return np.stack(predictions).astype(np.float32)


def _node_mean_nse(observed: np.ndarray, predicted: np.ndarray, day_mask: np.ndarray) -> float:
    scores: list[float] = []
    for j in range(observed.shape[1]):
        y = observed[day_mask, j]
        p = predicted[day_mask, j]
        score = nse_score(y, p)
        if np.isfinite(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else float("nan")


def monthly_totals_for_mask(
    daily_prediction: np.ndarray,
    monthly_observed: np.ndarray,
    dates: pd.DatetimeIndex,
    day_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    start = 0
    while start < len(dates):
        year, month = dates[start].year, dates[start].month
        end = start + 1
        while end < len(dates) and dates[end].year == year and dates[end].month == month:
            end += 1
        if np.all(day_mask[start:end]):
            obs = monthly_observed[end - 1]
            if np.isfinite(obs).any():
                pred_rows.append(daily_prediction[start:end].sum(axis=0))
                obs_rows.append(obs.copy())
        start = end
    if not pred_rows:
        shape = (0, daily_prediction.shape[1])
        return np.empty(shape, dtype=np.float32), np.empty(shape, dtype=np.float32)
    return np.stack(pred_rows).astype(np.float32), np.stack(obs_rows).astype(np.float32)


def validation_nse(
    bundle: SiteBundle,
    prediction: np.ndarray,
    validation_mask: np.ndarray,
    validation_daily_targets: np.ndarray,
) -> float:
    if bundle.supervision_mode == "monthly":
        pred_month, obs_month = monthly_totals_for_mask(
            prediction, bundle.obs_monthly_total, bundle.dates, validation_mask
        )
        # The research training path selected the DCM checkpoint using pooled
        # monthly-total NSE over all valid monitored month-node pairs.
        return nse_score(obs_month.reshape(-1), pred_month.reshape(-1))
    return _node_mean_nse(validation_daily_targets, prediction, validation_mask)


def save_test_outputs(
    bundle: SiteBundle,
    prediction: np.ndarray,
    test_mask: np.ndarray,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for t, date in enumerate(bundle.dates):
        for j, node_id in enumerate(bundle.node_ids):
            rows.append(
                {
                    "site": bundle.site,
                    "date": date,
                    "node_id": node_id,
                    "pred_m3_d": float(prediction[t, j]),
                    "observed_m3_d": (
                        float(bundle.obs_daily[t, j]) if np.isfinite(bundle.obs_daily[t, j]) else np.nan
                    ),
                    "split": "test" if test_mask[t] else "other",
                }
            )
    long_df = pd.DataFrame(rows)
    long_df.to_csv(out_dir / "all_nodes_daily_flow.csv", index=False)

    if bundle.supervision_mode == "monthly":
        pred_month, obs_month = monthly_totals_for_mask(
            prediction, bundle.obs_monthly_total, bundle.dates, test_mask
        )
        metrics = metrics_by_node(obs_month, pred_month, bundle.node_ids)
        metrics.to_csv(out_dir / "test_metrics_monthly.csv", index=False)

        sparse_obs = bundle.obs_daily[test_mask]
        sparse_pred = prediction[test_mask]
        if np.isfinite(sparse_obs).any():
            metrics_by_node(sparse_obs, sparse_pred, bundle.node_ids).to_csv(
                out_dir / "test_metrics_sparse_daily.csv", index=False
            )
    else:
        raw_obs = bundle.obs_daily[test_mask]
        raw_pred = prediction[test_mask]
        metrics = metrics_by_node(raw_obs, raw_pred, bundle.node_ids)
        metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    return metrics
