"""Main-model training. Model selection and final metrics use raw daily observations."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from data import load_site, default_splits, attach_dexing_obs_state, attach_obs_lags, SWAT_COLS
from model import PEGATGRU
from metrics import metrics_table, nse_score


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device_static(bundle, device):
    keys = [
        "area_m2", "c0", "vmax", "asurf", "qmine_base", "qintake_base", "q_recycle_base", "qcap",
        "is_s", "is_r", "is_t", "is_o", "is_surface_s", "is_mine_s", "is_intake_s",
    ]
    return {k: torch.tensor(getattr(bundle, k), dtype=torch.float32, device=device) for k in keys}


def standardize_series(X, train_mask):
    mu = X[train_mask].mean(axis=(0, 1), keepdims=True)
    sg = X[train_mask].std(axis=(0, 1), keepdims=True)
    sg = np.where(sg < 1e-6, 1.0, sg)
    return ((X - mu) / sg).astype(np.float32), mu.astype(np.float32), sg.astype(np.float32)


class MaskedLoss(nn.Module):
    """Huber loss on log1p flow plus a relative-flow term."""

    def __init__(self, per_gauge: bool = True):
        super().__init__()
        self.huber = nn.SmoothL1Loss(reduction="none")
        self.per_gauge = per_gauge

    def forward(self, pred, y, mask, gauge_w=None):
        m = mask
        if m.sum() < 1:
            return pred.new_zeros(())
        elem = self.huber(torch.log1p(pred.clamp_min(0)), torch.log1p(y.clamp_min(0)))
        scale = y.abs().clamp_min(50.0)
        elem = elem + 0.5 * self.huber(pred / scale, y / scale)
        if gauge_w is not None:
            elem = elem * gauge_w
        if self.per_gauge and pred.dim() == 1:
            # pred (N,), mask (N,) — mean over active gauges equally
            active = m > 0
            if active.sum() < 1:
                return pred.new_zeros(())
            return (elem * m)[active].mean()
        return (elem * m).sum() / m.sum().clamp_min(1.0)


def _qhat_feat(Q_prev: torch.Tensor, q_scale: float) -> torch.Tensor:
    """Normalize previous *model* prediction for GAT input (N,1). Not observed flow."""
    return (torch.log1p(Q_prev.clamp_min(0.0)) / max(q_scale, 1.0)).unsqueeze(-1)


def step_day(
    model, x_t, phys_t, V, h_gru, static_t, edges, Q_prev=None, q_scale=8.0,
    use_pred_lag=False, p_slow=None, slow_rho=0.7,
    obs_lag1=None, gauged_mask=None, use_obs_gate=False, obs_gate_max=0.9,
):
    edge_src, edge_dst, edge_alpha = edges
    if use_pred_lag:
        if Q_prev is None:
            Q_prev = torch.zeros(x_t.shape[0], device=x_t.device, dtype=x_t.dtype)
        x_t = torch.cat([x_t, _qhat_feat(Q_prev, q_scale)], dim=-1)
    xt = torch.cat([x_t.unsqueeze(0), static_t["_xstat"].unsqueeze(0)], dim=-1)
    emb = model.encode_step(xt)
    out, h_gru = model.gru(emb.squeeze(0).unsqueeze(1), h_gru)
    h_nodes = out.squeeze(1).unsqueeze(0)
    p = model.params_from_hidden(h_nodes, edge_src, edge_dst)
    # Slow-parameter inertia: operational parameters carry across days.
    slow_keys = ("k_mine", "k_intake", "k_rec", "r", "p", "eta", "q_ext")
    if p_slow is None or slow_rho <= 0:
        p_use = p
        p_slow_next = {k: p[k].detach() for k in slow_keys} if slow_rho > 0 else None
    else:
        rho = float(slow_rho)
        p_use = dict(p)
        p_slow_next = {}
        for k in slow_keys:
            blended = rho * p_slow[k] + (1.0 - rho) * p[k]
            p_use[k] = blended
            p_slow_next[k] = blended.detach()
    Q, V = model.balance(
        precip_mm=phys_t[:, 0].unsqueeze(0),
        snowmelt_mm=phys_t[:, 1].unsqueeze(0),
        pet_mm=phys_t[:, 2].unsqueeze(0),
        V=V,
        k_c=p["k_c"],
        k_snow=p["k_snow"],
        k_mine=p_use["k_mine"],
        k_intake=p_use["k_intake"],
        k_e=p["k_e"],
        r=p_use["r"],
        p=p_use["p"],
        eta=p_use["eta"],
        delta_alpha=p["delta_alpha"],
        q_intake_free=p["q_intake_free"],
        k_rec=p_use["k_rec"],
        k_gw=p["k_gw"],
        q_ext=p_use["q_ext"],
        gw_mm=phys_t[:, 3].unsqueeze(0),
        ops_scale=phys_t[:, 4].unsqueeze(0),
        pool_drive=phys_t[:, 5].unsqueeze(0),
        q_plan_recycle=phys_t[:, 6].unsqueeze(0),
        q_plan_treat=phys_t[:, 7].unsqueeze(0),
        area_m2=static_t["area_m2"],
        c0=static_t["c0"],
        vmax=static_t["vmax"],
        asurf=static_t["asurf"],
        qmine_base=static_t["qmine_base"],
        qintake_base=static_t["qintake_base"],
        q_recycle_base=static_t["q_recycle_base"],
        qcap=static_t["qcap"],
        is_s=static_t["is_s"],
        is_r=static_t["is_r"],
        is_t=static_t["is_t"],
        is_o=static_t["is_o"],
        is_surface_s=static_t["is_surface_s"],
        is_mine_s=static_t["is_mine_s"],
        is_intake_s=static_t["is_intake_s"],
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_alpha=edge_alpha,
    )
    Q = Q.squeeze(0)
    # End-to-end trained causal obs gate on gauged nodes only (not post-hoc assim)
    if use_obs_gate and obs_lag1 is not None and gauged_mask is not None:
        g = torch.sigmoid(model.obs_gate(h_nodes)).squeeze(0).squeeze(-1) # (N,)
        g = (g * gauged_mask * float(obs_gate_max)).clamp(0.0, float(obs_gate_max))
        Q = (1.0 - g) * Q + g * obs_lag1
    return Q, V, h_gru, p_slow_next


def run_series(model, X_std, phys, static_t, edges, Y, M, day_mask, criterion, device,
               train=False, optimizer=None, chunk=48, use_pred_lag=False, q_scale=8.0,
               gauge_w=None,
               Y_lag1=None, gauged_mask=None, use_obs_gate=False, slow_rho=0.0):
    """Sequential daily prediction and daily-flow supervision on training dates."""
    T, N, _ = X_std.shape
    V = model.initial_storage(static_t["vmax"], 1).to(device)
    h_gru = torch.zeros(1, N, model.gru.hidden_size, device=device)
    Q_prev = torch.zeros(N, device=device)
    p_slow = None
    preds, losses = [], []
    model.train(mode=train)
    gw = None
    if gauge_w is not None:
        gw = torch.as_tensor(gauge_w, dtype=torch.float32, device=device)
    gmask = None
    if gauged_mask is not None:
        gmask = torch.as_tensor(gauged_mask, dtype=torch.float32, device=device)

    bounds = [(t, min(T, t + chunk)) for t in range(0, T, chunk)]

    for t0, t1 in bounds:
        if train and not np.any(day_mask[t0:]): break
        if train:
            if t0 > 0:
                V = V.detach()
                h_gru = h_gru.detach()
                Q_prev = Q_prev.detach()
                if p_slow is not None:
                    p_slow = {k: v.detach() for k, v in p_slow.items()}
            optimizer.zero_grad(set_to_none=True)
        chunk_preds = []
        chunk_loss = None
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for t in range(t0, t1):
                x_t = torch.as_tensor(X_std[t], dtype=torch.float32, device=device)
                phys_t = torch.as_tensor(phys[t], dtype=torch.float32, device=device)
                ol1 = None
                if Y_lag1 is not None:
                    ol1 = torch.as_tensor(Y_lag1[t], dtype=torch.float32, device=device)
                Q, V, h_gru, p_slow = step_day(
                    model, x_t, phys_t, V, h_gru, static_t, edges,
                    Q_prev=Q_prev, q_scale=q_scale, use_pred_lag=use_pred_lag,
                    p_slow=p_slow, slow_rho=slow_rho,
                    obs_lag1=ol1, gauged_mask=gmask,
                    use_obs_gate=use_obs_gate, obs_gate_max=0.92,
                )
                if not torch.isfinite(Q).all() or not torch.isfinite(V).all(): raise FloatingPointError("Nonfinite model state")
                Q_prev = Q.detach()
                chunk_preds.append(Q)
                if day_mask[t] and float(M[t].sum()) > 0:
                    y_t = torch.as_tensor(np.nan_to_num(Y[t], nan=0.0), dtype=torch.float32, device=device)
                    m_t = torch.as_tensor(M[t], dtype=torch.float32, device=device)
                    if float(m_t.sum()) > 0:
                        lt = criterion(Q, y_t, m_t, gauge_w=gw)
                        chunk_loss = lt if chunk_loss is None else chunk_loss + lt
        pred_chunk = torch.stack(chunk_preds, dim=0)
        preds.append(pred_chunk.detach().cpu().numpy())
        if train and chunk_loss is not None:
            loss = chunk_loss
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
    return np.concatenate(preds, axis=0), float(np.mean(losses)) if losses else float("nan")


def daily_selection_nse(P, Y, M, node_ids, observed, day_idx, node_weights: dict[str, float] | None = None):
    """Selection metric: weighted mean NSE over gauges."""
    scores, weights = [], []
    for j, nid in enumerate(node_ids):
        if nid not in observed:
            continue
        sel = (M[day_idx, j] > 0) & np.isfinite(Y[day_idx, j])
        if sel.sum() < 5:
            continue
        s = nse_score(Y[day_idx][sel, j], P[day_idx][sel, j])
        if not np.isfinite(s):
            continue
        w = float((node_weights or {}).get(nid, 1.0))
        scores.append(s * w)
        weights.append(w)
    if not weights:
        return float("nan")
    return float(np.sum(scores) / np.sum(weights))


def parse_weights(spec, node_ids):
    weights = {}
    for entry in spec.split(','):
        if not entry.strip():
            continue
        node, value = entry.split('=')
        node, value = node.strip(), float(value)
        if node not in node_ids or not np.isfinite(value) or value <= 0:
            raise ValueError('Gauge weights require known_node=positive_number')
        if node in weights:
            raise ValueError('Duplicate gauge weight')
        weights[node] = value
    return weights


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def train_site(site, args):
    """Train/select on chronological splits, or replay one saved checkpoint."""
    seed_everything(args.seed)
    device = torch.device(args.device)
    saved = None
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if saved.get('format_version') != 2 or saved['site'] != site:
            raise ValueError('Unsupported checkpoint or mismatched site')
    config_keys = ('graph_dim', 'hidden_dim', 'dropout', 'balance_iters', 'chunk',
                   'obs_lag', 'obs_gate',
                   'gauge_weights', 'select_weights', 'o3_weight')
    cfg = saved['config'] if saved else {k: getattr(args, k) for k in config_keys}
    bundle = load_site(site, Path(args.data_root))
    dexing = site == 'dexing'
    if dexing:
        bundle = attach_dexing_obs_state(bundle)
    lags = tuple(int(x.strip()) for x in cfg['obs_lag'].split(',') if x.strip() and x.strip() != '0')
    if any(x <= 0 for x in lags) or len(set(lags)) != len(lags):
        raise ValueError('Observation lags must be distinct positive integers')
    if dexing and (lags or cfg['obs_gate']):
        raise ValueError('Daily observation-lag/gate options apply to Jiama only')
    if lags:
        bundle = attach_obs_lags(bundle, lags=lags)
    dates, nodes = bundle.dates, bundle.node_ids
    if 'ALL_OBSERVED' in nodes:
        raise ValueError('ALL_OBSERVED is reserved for pooled metrics')
    ranges = default_splits(site)
    split_masks = {s: np.asarray((dates >= ranges[s+'_start']) & (dates <= ranges[s+'_end']))
                   for s in ('train', 'valid', 'test')}
    if not all(m.any() for m in split_masks.values()):
        raise ValueError('Input dates must cover training, validation and test periods')
    train_m, valid_m, test_m = (split_masks[s] for s in ('train', 'valid', 'test'))
    raw = bundle.Y.copy()
    if np.isinf(raw).any() or (raw[np.isfinite(raw)] < 0).any():
        raise ValueError('Observed daily flow must be nonnegative or missing')
    eval_mask = np.isfinite(raw).astype(np.float32)
    target = raw.copy()
    target_mask = np.isfinite(target).astype(np.float32)
    if not np.isfinite(target[train_m]).any():
        raise ValueError('No observed training labels')
    select_w = parse_weights(cfg['select_weights'], nodes)
    valid_idx = np.flatnonzero(valid_m)
    if not saved and not np.isfinite(daily_selection_nse(raw, raw, eval_mask, nodes, bundle.observed_nodes, valid_idx, select_w)):
        raise ValueError('Daily validation NSE requires at least five observations with nonzero variance at one monitored node')
    if not np.isfinite(raw[test_m]).any():
        raise ValueError('No daily observations in the test period')

    phys = np.stack([bundle.precip_mm, bundle.snowmelt_mm, bundle.pet_mm,
                     bundle.X_dyn[:, :, SWAT_COLS.index('GW_Qmm')], bundle.ops_scale,
                     bundle.pool_drive, bundle.q_plan_recycle, bundle.q_plan_treat], axis=-1).astype(np.float32)
    if not all(np.isfinite(a).all() for a in (bundle.X_dyn, bundle.X_static, phys)):
        raise ValueError('Nonfinite model input; check private forcing and operating records')
    if saved:
        if nodes != saved['node_ids'] or bundle.feature_names != saved['feature_names'] or ranges != saved['splits']:
            raise ValueError('Checkpoint node order, features or time splits differ from inputs')
        for key in ('edge_src', 'edge_dst', 'edge_alpha'):
            if not np.array_equal(getattr(bundle, key), saved[key].numpy()):
                raise ValueError('Checkpoint engineering graph differs from inputs')
        mu, sg, smu, ssig = (saved[k].numpy() for k in ('dyn_mean', 'dyn_std', 'static_mean', 'static_std'))
        X_std = ((bundle.X_dyn-mu)/sg).astype(np.float32)
        q_scale = saved['q_scale']
    else:
        X_std, mu, sg = standardize_series(bundle.X_dyn, train_m)
        smu = bundle.X_static.mean(0, keepdims=True)
        ssig = bundle.X_static.std(0, keepdims=True)
        ssig = np.where(ssig < 1e-6, 1.0, ssig).astype(np.float32)
        scale_data = target[train_m]
        finite = scale_data[np.isfinite(scale_data)]
        q_scale = max(float(np.log1p(finite.mean())), 1.0) if finite.size else 8.0
    xs_std = ((bundle.X_static-smu)/ssig).astype(np.float32)
    static = to_device_static(bundle, device)
    if not all(torch.isfinite(v).all() for v in static.values()):
        raise ValueError('Nonfinite physical node property')
    static['_xstat'] = torch.as_tensor(xs_std, device=device)
    edges = (torch.as_tensor(bundle.edge_src, dtype=torch.long, device=device),
             torch.as_tensor(bundle.edge_dst, dtype=torch.long, device=device),
             torch.as_tensor(bundle.edge_alpha, dtype=torch.float32, device=device))
    model = PEGATGRU(X_std.shape[-1]+1, xs_std.shape[-1], len(nodes), len(bundle.edge_src),
                    graph_dim=cfg['graph_dim'], hidden_dim=cfg['hidden_dim'], dropout=cfg['dropout'],
                    balance_iters=cfg['balance_iters']).to(device)
    model.set_graph(edges[0], edges[1])
    if saved:
        model.load_state_dict(saved['model_state'])
    gauge_weights = np.ones(len(nodes), dtype=np.float32)
    gw = parse_weights(cfg['gauge_weights'], nodes)
    if dexing and 'O3' in nodes and 'O3' not in gw:
        gw['O3'] = cfg['o3_weight']
    for node, weight in gw.items():
        gauge_weights[nodes.index(node)] = weight
    use_gate = not dexing and bool(lags or cfg['obs_gate'])
    lag1 = np.zeros_like(raw)
    gauged = np.zeros(len(nodes), dtype=np.float32)
    for node in bundle.observed_nodes:
        j = nodes.index(node)
        gauged[j] = 1
        # The observation gate uses only values available before the predicted day.
        series = pd.Series(target[:, j]).ffill().fillna(0).to_numpy(dtype=np.float32)
        lag1[1:, j] = series[:-1]
    criterion = MaskedLoss()
    common = dict(chunk=cfg['chunk'], use_pred_lag=True, q_scale=q_scale,
                  gauge_w=gauge_weights, gauged_mask=gauged, use_obs_gate=use_gate,
                  slow_rho=0.75 if dexing else 0.0)

    def replay(end):
        # No evaluation targets enter loss or same-day prediction.
        return run_series(model, X_std[:end], phys[:end], static, edges, target[:end], target_mask[:end],
                          np.zeros(end, dtype=bool), criterion, device,
                          Y_lag1=lag1[:end] if use_gate else None, **common)[0]

    out = Path(args.out_dir)/site
    out.mkdir(parents=True, exist_ok=True)
    best_path = out/'best_hycongt.pt'
    history = []
    if not saved:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best, stale = -float('inf'), 0
        for epoch in range(1, args.epochs+1):
            _, loss = run_series(model, X_std, phys, static, edges, target, target_mask, train_m,
                                 criterion, device, train=True, optimizer=optimizer,
                                 Y_lag1=lag1 if use_gate else None, **common)
            val_pred = replay(int(valid_idx[-1])+1)
            score = daily_selection_nse(val_pred, raw[:len(val_pred)], eval_mask[:len(val_pred)],
                                        nodes, bundle.observed_nodes, valid_idx, select_w)
            if not np.isfinite(score):
                raise FloatingPointError('Nonfinite daily validation NSE')
            history.append({'epoch': epoch, 'train_loss': loss, 'valid_daily_NSE': score})
            print(f'{site}: epoch {epoch}, loss={loss:.5g}, validation daily NSE={score:.5g}', flush=True)
            if score > best:
                best, stale = score, 0
                checkpoint = dict(format_version=2, site=site, config=cfg, splits=ranges,
                                  epoch=epoch, valid_daily_NSE=score, q_scale=q_scale,
                                  node_ids=nodes, feature_names=bundle.feature_names,
                                  model_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                  seed=args.seed, lr=args.lr, weight_decay=args.weight_decay)
                for name, array in [('dyn_mean', mu), ('dyn_std', sg), ('static_mean', smu), ('static_std', ssig),
                                    ('edge_src', bundle.edge_src), ('edge_dst', bundle.edge_dst), ('edge_alpha', bundle.edge_alpha)]:
                    checkpoint[name] = torch.as_tensor(array).clone()
                temp = best_path.with_suffix('.tmp')
                torch.save(checkpoint, temp)
                temp.replace(best_path)
            else:
                stale += 1
            if stale >= args.patience:
                break
        saved = torch.load(best_path, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model_state'])
        pd.DataFrame(history).to_csv(out/'training_history.csv', index=False)
    predictions = replay(len(dates))
    split_labels = np.full(len(dates), 'outside_split', dtype=object)
    for split, mask in split_masks.items():
        split_labels[mask] = split
    table = pd.DataFrame({'date': np.repeat(dates.strftime('%Y-%m-%d'), len(nodes)),
                          'node_id': np.tile(nodes, len(dates)),
                          'observed_m3_d': raw.ravel(), 'pred_m3_d': predictions.ravel(),
                          'split': np.repeat(split_labels, len(nodes))})
    table.to_csv(out/'all_nodes_daily_flow.csv', index=False)
    tables = []
    for split in split_masks:
        result = metrics_table(table.loc[table['split'] == split])
        result.insert(0, 'split', split)
        tables.append(result)
    metrics = pd.concat(tables, ignore_index=True)
    metrics.to_csv(out/'daily_metrics.csv', index=False)
    metadata = dict(site=site, selected_epoch=saved['epoch'], selected_valid_daily_NSE=saved['valid_daily_NSE'],
                    config=cfg, splits=ranges, daily_reference='raw measured daily flow',
                    selection='weighted mean gauge daily NSE; minimum five observations per gauge',
                    supervision=bundle.supervise_mode, observation_gate=use_gate,
                    historical_observations_as_inputs=bool(dexing or lags or use_gate),
                    evaluation_protocol='chronological reconstruction with causal observation inputs when enabled',
                    seed=saved['seed'], lr=saved['lr'], weight_decay=saved['weight_decay'],
                    torch_version=str(torch.__version__), numpy_version=np.__version__, pandas_version=pd.__version__)
    (out/'run_metadata.json').write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False), encoding='utf-8')
    print(f'{site}: saved daily flow and raw-observation metrics to {out}', flush=True)
    return table, metrics


def main():
    p = argparse.ArgumentParser(description='HyConGT main model: daily validation selection and raw daily evaluation.')
    p.add_argument('--site', choices=('dexing', 'jiama', 'all'), default='all')
    p.add_argument('--data_root', required=True, help='Private input directory; see DATA_SCHEMA.md')
    p.add_argument('--out_dir', default='private_outputs')
    p.add_argument('--checkpoint', help='Replay a saved main-model checkpoint; requires one explicit site and the full input history')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--graph_dim', type=int, default=64)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--balance_iters', type=int, default=3)
    p.add_argument('--chunk', type=int, default=40)
    p.add_argument('--obs_lag', default='0', help='Jiama positive day lags, e.g. 1,3,7; enables the original trained observation gate')
    p.add_argument('--obs_gate', action='store_true', help='Enable the original Jiama causal observation gate independently of lag features')
    p.add_argument('--o3_weight', type=float, default=2.5, help='Original Dexing O3 training weight')
    p.add_argument('--gauge_weights', default='', help='Training weights, e.g. O1=1,O3=2.5')
    p.add_argument('--select_weights', default='', help='Daily validation selection weights; equal gauge weights by default')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if any(getattr(args, k) < 1 for k in ('epochs','patience','graph_dim','hidden_dim','balance_iters','chunk','threads')):
        p.error('Integer sizes, iterations and patience must be positive')
    values = [args.lr, args.weight_decay, args.dropout, args.o3_weight]
    if not np.isfinite(values).all() or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.dropout < 1 or args.o3_weight <= 0:
        p.error('Invalid learning, dropout or loss-weight setting')
    if args.graph_dim % 4:
        p.error('graph_dim must be divisible by four attention heads')
    if args.checkpoint and args.site == 'all':
        p.error('Checkpoint replay requires --site dexing or --site jiama')
    torch.set_num_threads(args.threads)
    for site in ('dexing', 'jiama') if args.site == 'all' else (args.site,):
        train_site(site, args)


if __name__ == '__main__':
    main()
