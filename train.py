from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hycongt.data import (
    load_prepared_site,
    split_masks,
    standardize_dynamic,
    standardize_static,
)
from hycongt.model import HyConGT
from hycongt.training import (
    CompositeHuberLoss,
    flow_log_scale,
    make_physics_dynamic,
    make_static_tensors,
    rollout,
    save_test_outputs,
    trailing_mean_targets,
    train_one_epoch,
    validation_nse,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the public HyConGT main model.")
    parser.add_argument("--data", required=True, help="Prepared confidential NPZ input.")
    parser.add_argument("--out", required=True, help="Output directory. Keep it out of Git.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--balance-iters", type=int, default=3)
    parser.add_argument("--delta-alpha-max", type=float, default=0.25)
    parser.add_argument("--chunk-days", type=int, default=40)
    parser.add_argument("--slow-rho", type=float, default=0.75)
    parser.add_argument("--lambda-month", type=float, default=1.0)
    parser.add_argument("--lambda-day", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_prepared_site(args.data)
    train_mask, valid_mask, test_mask = split_masks(bundle)
    x_dynamic, dyn_mu, dyn_sigma = standardize_dynamic(bundle.x_dynamic, train_mask)
    x_static, static_mu, static_sigma = standardize_static(bundle.x_static)
    physics = make_physics_dynamic(bundle)

    if bundle.supervision_mode == "daily":
        train_targets = trailing_mean_targets(bundle.obs_daily, window=7)
        validation_targets = train_targets
        lambda_day = 1.0
    else:
        train_targets = bundle.obs_daily.copy()
        validation_targets = bundle.obs_daily.copy()
        lambda_day = args.lambda_day

    q_scale = flow_log_scale(bundle, train_mask)
    model = HyConGT(
        dynamic_dim=x_dynamic.shape[-1] + 1,
        static_dim=x_static.shape[-1],
        n_nodes=bundle.n_nodes,
        graph_dim=args.graph_dim,
        hidden_dim=args.hidden_dim,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        balance_iters=args.balance_iters,
        delta_alpha_max=args.delta_alpha_max,
    ).to(device)

    edge_src = torch.as_tensor(bundle.edge_src, dtype=torch.long, device=device)
    edge_dst = torch.as_tensor(bundle.edge_dst, dtype=torch.long, device=device)
    edge_alpha = torch.as_tensor(bundle.edge_alpha, dtype=torch.float32, device=device)
    model.set_graph(edge_src, edge_dst)
    edges = (edge_src, edge_dst, edge_alpha)
    static = make_static_tensors(bundle, x_static, device)

    criterion = CompositeHuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = out_dir / "best_hycongt.pt"
    best_nse = -np.inf
    patience_left = args.patience
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            x_dynamic=x_dynamic,
            physics=physics,
            static=static,
            edges=edges,
            bundle=bundle,
            train_mask=train_mask,
            train_targets=train_targets,
            q_scale=q_scale,
            slow_rho=args.slow_rho,
            chunk_days=args.chunk_days,
            lambda_month=args.lambda_month,
            lambda_day=lambda_day,
            device=device,
        )
        prediction = rollout(
            model, x_dynamic, physics, static, edges, q_scale, args.slow_rho, device
        )
        valid_nse = validation_nse(bundle, prediction, valid_mask, validation_targets)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_NSE": valid_nse})
        print(f"epoch {epoch:03d} train_loss={train_loss:.6f} valid_NSE={valid_nse:.6f}")

        if np.isfinite(valid_nse) and valid_nse > best_nse + 1e-4:
            best_nse = valid_nse
            patience_left = args.patience
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "dynamic_mean": dyn_mu,
                    "dynamic_std": dyn_sigma,
                    "static_mean": static_mu,
                    "static_std": static_sigma,
                    "q_scale": q_scale,
                    "valid_NSE": valid_nse,
                    "site": bundle.site,
                    "node_ids": bundle.node_ids,
                },
                best_path,
            )
        else:
            patience_left -= 1
            if epoch >= 8 and patience_left <= 0:
                break

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    if not best_path.exists():
        raise RuntimeError("No finite validation NSE was available for checkpoint selection.")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    prediction = rollout(model, x_dynamic, physics, static, edges, q_scale, args.slow_rho, device)
    metrics = save_test_outputs(bundle, prediction, test_mask, out_dir)

    summary = {
        "site": bundle.site,
        "selection_metric": "validation_NSE",
        "best_validation_NSE": float(best_nse),
        "supervision_mode": bundle.supervision_mode,
        "test_metrics": metrics.to_dict("records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
