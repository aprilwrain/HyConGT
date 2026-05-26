from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset


DYNAMIC_FEATURE_COLUMNS = [
    "PRECIPmm",
    "SNOWMELTmm",
    "ETmm",
    "SWmm",
    "PERCmm",
    "SURQmm",
    "GW_Qmm",
    "SYLDt_ha",
    "LAT_Qmm",
    "sim_outflow",
]

STATIC_FEATURE_COLUMNS = [
    "SUB_area_m2",
    "area_m2",
    "yihongkoubiaogao_m",
    "shuimianmianji_m2",
    "youxiaokurong_m3",
    "zuidabengpainengli_m3/h",
    "jiangyuhuishuimianji_m2",
    "jingliuxishu",
    "S",
    "R",
    "T",
    "O",
]

CALENDAR_FEATURE_COLUMNS = ["month_sin", "month_cos", "doy_sin", "doy_cos"]
MODEL_NAME = "HyConGAT"


@dataclass
class Config:
    node_csv: str = "./data/node.csv"
    edge_csv: str = "./data/edges.csv"
    daily_csv: str = "./outputs/merge_node_sub_day_real_swat_outflow_phys.csv"
    out_dir: str = "./hycongat_outputs"

    seq_len: int = 60
    pred_horizon: int = 1
    hidden_dim: int = 128
    graph_out_dim: int = 128
    graph_layers: int = 3
    gru_layers: int = 1
    gat_heads: int = 4
    dropout: float = 0.15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 200
    patience: int = 30
    grad_clip: float = 1.0

    train_start: str = "2020-01-01"
    train_end: str = "2024-06-30"
    valid_start: str = "2024-07-01"
    valid_end: str = "2024-09-30"
    test_start: str = "2025-01-01"
    test_end: str = "2025-07-31"

    add_reverse_edges: bool = True
    add_self_loops: bool = True
    use_residual_graph: bool = True
    add_target_id_embedding: bool = True
    target_log1p: bool = True

    loss_extreme_weight_lambda: float = 4.0
    huber_delta: float = 1.0
    target_loss_weights: str = "O1:2.0,O2:2.0,O3:0.8"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> Config:
    cfg = Config()
    parser = argparse.ArgumentParser(description=f"Train and evaluate {MODEL_NAME}.")

    for field_info in fields(cfg):
        name = field_info.name
        default = getattr(cfg, name)
        arg_name = f"--{name}"
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=str_to_bool, default=default)
        elif isinstance(default, int):
            parser.add_argument(arg_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(arg_name, type=float, default=default)
        else:
            parser.add_argument(arg_name, default=default)

    args = parser.parse_args()
    for field_info in fields(cfg):
        setattr(cfg, field_info.name, getattr(args, field_info.name))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is not available. Falling back to CPU.")
        return "cpu"
    return device_name


def parse_yyyydoy(value: int) -> pd.Timestamp:
    text = str(int(value))
    if len(text) != 7:
        raise ValueError(f"YYYYDDD must contain 7 digits, got {value!r}.")
    return pd.Timestamp(year=int(text[:4]), month=1, day=1) + pd.Timedelta(days=int(text[4:]) - 1)


def safe_log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(values, 0.0, None))


def inverse_safe_log1p(values: np.ndarray) -> np.ndarray:
    return np.expm1(values)


def parse_target_weights(raw: str, target_nodes: List[str]) -> Dict[str, float]:
    weights = {node: 1.0 for node in target_nodes}
    if not raw:
        return weights

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Use --target_loss_weights in the form 'O1:2.0,O2:2.0,O3:0.8'.")
        key, value = item.split(":", 1)
        weights[key.strip()] = float(value)
    return {node: weights.get(node, 1.0) for node in target_nodes}


def validate_columns(frame: pd.DataFrame, required_columns: List[str], file_label: str) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_label}: {missing}")


def build_adjacency(
    node_ids: List[str],
    edges: pd.DataFrame,
    add_reverse_edges: bool,
    add_self_loops: bool,
) -> np.ndarray:
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=np.float32)

    for _, row in edges.iterrows():
        source = row["from_id"]
        destination = row["to_id"]
        if source not in node_to_index or destination not in node_to_index:
            continue
        source_index = node_to_index[source]
        destination_index = node_to_index[destination]
        weight = float(row["weight_alpha"])
        adjacency[source_index, destination_index] = max(adjacency[source_index, destination_index], weight)
        if add_reverse_edges:
            adjacency[destination_index, source_index] = max(adjacency[destination_index, source_index], weight)

    if add_self_loops:
        np.fill_diagonal(adjacency, np.maximum(np.diag(adjacency), 1.0))
    return adjacency


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    inverse_sqrt_degree = np.diag(np.power(np.clip(degree, 1e-12, None), -0.5))
    return inverse_sqrt_degree @ adjacency @ inverse_sqrt_degree


def build_static_features(nodes: pd.DataFrame, node_ids: List[str]) -> np.ndarray:
    validate_columns(nodes, ["node_id"] + STATIC_FEATURE_COLUMNS, "node_csv")
    node_table = nodes.set_index("node_id").loc[node_ids].copy()
    static_features = node_table[STATIC_FEATURE_COLUMNS].fillna(0.0).astype(np.float32).values

    for column in ["S", "R", "T", "O"]:
        if column in STATIC_FEATURE_COLUMNS:
            column_index = STATIC_FEATURE_COLUMNS.index(column)
            static_features[:, column_index] = np.log1p(np.clip(static_features[:, column_index], 0.0, None))
    return static_features


def add_calendar_features(daily: pd.DataFrame) -> pd.DataFrame:
    output = daily.copy()
    output["date"] = output["YYYYDDD"].apply(parse_yyyydoy)
    month = output["date"].dt.month
    day_of_year = output["date"].dt.dayofyear
    output["month_sin"] = np.sin(2.0 * math.pi * month / 12.0)
    output["month_cos"] = np.cos(2.0 * math.pi * month / 12.0)
    output["doy_sin"] = np.sin(2.0 * math.pi * day_of_year / 366.0)
    output["doy_cos"] = np.cos(2.0 * math.pi * day_of_year / 366.0)
    return output


def build_dynamic_tensor(
    daily: pd.DataFrame,
    node_ids: List[str],
    target_nodes: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp], List[str]]:
    required_columns = ["YYYYDDD", "node_id", "real"]
    validate_columns(daily, required_columns, "daily_csv")

    daily = add_calendar_features(daily)
    available_dynamic_columns = [column for column in DYNAMIC_FEATURE_COLUMNS if column in daily.columns]
    missing_dynamic_columns = sorted(set(DYNAMIC_FEATURE_COLUMNS) - set(available_dynamic_columns))
    if missing_dynamic_columns:
        print(f"Skipped missing dynamic feature columns: {missing_dynamic_columns}")

    feature_columns = available_dynamic_columns + CALENDAR_FEATURE_COLUMNS
    dates = sorted(daily["date"].unique())
    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    date_to_index = {date: index for index, date in enumerate(dates)}
    target_to_index = {node_id: index for index, node_id in enumerate(target_nodes)}

    dynamic_tensor = np.zeros((len(dates), len(node_ids), len(feature_columns)), dtype=np.float32)
    targets = np.full((len(dates), len(target_nodes)), np.nan, dtype=np.float32)

    for _, row in daily.iterrows():
        node_id = row["node_id"]
        if node_id not in node_to_index:
            continue
        time_index = date_to_index[row["date"]]
        node_index = node_to_index[node_id]
        dynamic_tensor[time_index, node_index, :] = np.array(
            [float(row[column]) if pd.notna(row.get(column, np.nan)) else 0.0 for column in feature_columns],
            dtype=np.float32,
        )
        if node_id in target_to_index and pd.notna(row["real"]):
            targets[time_index, target_to_index[node_id]] = float(row["real"])

    return dynamic_tensor, targets, list(dates), feature_columns


def build_sequence_samples(
    dynamic_tensor: np.ndarray,
    targets: np.ndarray,
    static_features: np.ndarray,
    dates: List[pd.Timestamp],
    seq_len: int,
    pred_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    static_block = np.repeat(static_features[None, :, :], seq_len, axis=0)
    all_inputs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    target_dates: List[pd.Timestamp] = []
    dates_index = pd.to_datetime(pd.Index(dates))
    skipped_non_contiguous = 0

    for time_index in range(seq_len, len(dates) - pred_horizon + 1):
        target_index = time_index + pred_horizon - 1
        window_dates = dates_index[time_index - seq_len : target_index + 1]
        if len(window_dates) >= 2:
            day_deltas = np.diff(window_dates.values).astype("timedelta64[D]").astype(int)
            if not np.all(day_deltas == 1):
                skipped_non_contiguous += 1
                continue

        target_values = targets[target_index].copy()
        if np.isnan(target_values).any():
            continue

        dynamic_window = dynamic_tensor[time_index - seq_len : time_index].copy()
        all_inputs.append(np.concatenate([dynamic_window, static_block], axis=-1))
        all_targets.append(target_values)
        target_dates.append(dates[target_index])

    if skipped_non_contiguous:
        print(f"Skipped {skipped_non_contiguous} samples with non-contiguous dates.")
    if not all_inputs:
        raise ValueError("No valid sequence samples were generated. Check dates, missing targets, and seq_len.")

    return (
        np.stack(all_inputs).astype(np.float32),
        np.stack(all_targets).astype(np.float32),
        np.array(target_dates),
    )


def split_by_target_date(target_dates: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(pd.Index(target_dates))
    valid_mask = (dates >= pd.Timestamp(cfg.valid_start)) & (dates <= pd.Timestamp(cfg.valid_end))
    train_mask = (dates >= pd.Timestamp(cfg.train_start)) & (dates <= pd.Timestamp(cfg.train_end)) & (~valid_mask)
    test_mask = (dates >= pd.Timestamp(cfg.test_start)) & (dates <= pd.Timestamp(cfg.test_end))

    train_idx = np.where(train_mask)[0]
    valid_idx = np.where(valid_mask)[0]
    test_idx = np.where(test_mask)[0]

    if len(train_idx) == 0:
        raise ValueError(f"No training samples in {cfg.train_start} to {cfg.train_end}.")
    if len(valid_idx) == 0:
        raise ValueError(f"No validation samples in {cfg.valid_start} to {cfg.valid_end}.")
    if len(test_idx) == 0:
        raise ValueError(f"No test samples in {cfg.test_start} to {cfg.test_end}.")

    def describe(indices: np.ndarray) -> str:
        return f"{dates[indices].min().date()} to {dates[indices].max().date()} ({len(indices)} samples)"

    print("Date split")
    print(f"  train: {describe(train_idx)}")
    print(f"  valid: {describe(valid_idx)}")
    print(f"  test : {describe(test_idx)}")
    return train_idx, valid_idx, test_idx


def standardize_by_training_set(inputs: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = inputs[train_idx]
    mean = train_values.reshape(-1, train_values.shape[-1]).mean(axis=0)
    std = train_values.reshape(-1, train_values.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (inputs - mean) / std, mean, std


class SequenceDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, indices: np.ndarray) -> None:
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Tuple[torch.Tensor, torch.Tensor]:
        index = self.indices[position]
        return self.inputs[index], self.targets[index]


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.projection = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_source = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_target = nn.Parameter(torch.empty(heads, out_dim))
        self.output_projection = nn.Linear(out_dim * heads, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.attn_source)
        nn.init.xavier_uniform_(self.attn_target)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = node_features.shape
        projected = self.projection(node_features).view(batch_size, node_count, self.heads, self.out_dim)
        source_scores = (projected * self.attn_source).sum(dim=-1)
        target_scores = (projected * self.attn_target).sum(dim=-1)
        attention_logits = self.leaky_relu(source_scores.unsqueeze(2) + target_scores.unsqueeze(1))

        edge_bias = torch.where(
            adjacency > 0,
            torch.log(adjacency + 1.0),
            torch.full_like(adjacency, -1e9),
        )
        attention_logits = attention_logits + edge_bias.unsqueeze(0).unsqueeze(-1)
        attention = torch.softmax(attention_logits, dim=2)
        attention = self.dropout(attention)

        aggregated = torch.matmul(
            attention.permute(0, 3, 1, 2),
            projected.permute(0, 2, 1, 3),
        )
        aggregated = aggregated.permute(0, 2, 1, 3).contiguous()
        aggregated = aggregated.view(batch_size, node_count, self.heads * self.out_dim)
        return self.output_projection(aggregated)


class GraphAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float, use_residual: bool) -> None:
        super().__init__()
        self.attention = GraphAttentionLayer(in_dim, out_dim, heads, dropout)
        self.residual = nn.Linear(in_dim, out_dim) if use_residual else None
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.attention(node_features, adjacency)
        if self.residual is not None:
            hidden = hidden + self.residual(node_features)
        hidden = self.norm(hidden)
        hidden = self.activation(hidden)
        return self.dropout(hidden)


class HyConGAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        graph_out_dim: int,
        hidden_dim: int,
        target_count: int,
        graph_layers: int,
        gat_heads: int,
        gru_layers: int,
        dropout: float,
        use_residual_graph: bool,
        add_target_id_embedding: bool,
    ) -> None:
        super().__init__()
        layer_dims = [in_dim] + [graph_out_dim] * graph_layers
        self.graph_blocks = nn.ModuleList(
            [
                GraphAttentionBlock(
                    layer_dims[i],
                    layer_dims[i + 1],
                    gat_heads,
                    dropout,
                    use_residual_graph,
                )
                for i in range(graph_layers)
            ]
        )

        self.add_target_id_embedding = add_target_id_embedding
        if add_target_id_embedding:
            self.target_embedding = nn.Embedding(target_count, 8)
            gru_input_dim = graph_out_dim + 8
        else:
            self.target_embedding = None
            gru_input_dim = graph_out_dim

        self.gru = nn.GRU(
            gru_input_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_sequence: torch.Tensor, adjacency: torch.Tensor, target_node_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = input_sequence.shape
        hidden_states = []

        for time_step in range(seq_len):
            node_state = input_sequence[:, time_step]
            for block in self.graph_blocks:
                node_state = block(node_state, adjacency)
            hidden_states.append(node_state.unsqueeze(1))

        target_states = torch.cat(hidden_states, dim=1)[:, :, target_node_indices, :]
        target_count = target_states.shape[2]

        if self.add_target_id_embedding and self.target_embedding is not None:
            target_ids = torch.arange(target_count, device=target_states.device)
            embeddings = self.target_embedding(target_ids)
            embeddings = embeddings.view(1, 1, target_count, -1).expand(batch_size, seq_len, target_count, -1)
            target_states = torch.cat([target_states, embeddings], dim=-1)

        gru_input = target_states.permute(0, 2, 1, 3).contiguous().view(batch_size * target_count, seq_len, -1)
        gru_output, _ = self.gru(gru_input)
        last_state = gru_output[:, -1, :]
        predictions = self.regressor(last_state).view(batch_size, target_count)
        return predictions


class FrequencyWeightedHuberLoss(nn.Module):
    def __init__(
        self,
        node_weights: torch.Tensor,
        training_targets: np.ndarray,
        extreme_weight_lambda: float,
        delta: float,
        n_bins: int = 20,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.extreme_weight_lambda = extreme_weight_lambda
        self.n_bins = n_bins
        self.target_count = training_targets.shape[1]
        self.register_buffer("node_weights", node_weights)

        for target_index in range(self.target_count):
            values = training_targets[:, target_index]
            values = values[~np.isnan(values)]
            counts, edges = np.histogram(values, bins=n_bins)
            frequencies = counts.astype(float) + 1.0
            weights = 1.0 + extreme_weight_lambda * np.clip(1.0 - frequencies / frequencies.max(), 0.0, 1.0)
            self.register_buffer(f"bin_edges_{target_index}", torch.tensor(edges, dtype=torch.float32))
            self.register_buffer(f"bin_weights_{target_index}", torch.tensor(weights, dtype=torch.float32))

    def sample_weights(self, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(targets)
        for target_index in range(self.target_count):
            edges = getattr(self, f"bin_edges_{target_index}").to(targets.device)
            bin_weights = getattr(self, f"bin_weights_{target_index}").to(targets.device)
            bin_index = torch.bucketize(targets[:, target_index], edges[1:-1]).clamp(0, self.n_bins - 1)
            weights[:, target_index] = bin_weights[bin_index]
        return weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        absolute_error = torch.abs(predictions - targets)
        quadratic_part = torch.clamp(absolute_error, max=self.delta)
        huber = 0.5 * quadratic_part**2 + self.delta * (absolute_error - quadratic_part)
        weights = self.sample_weights(targets) * self.node_weights.view(1, -1)
        return (huber * weights).mean()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    denominator = np.clip(np.abs(y_true), 1e-6, None)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0),
    }


def evaluate(
    model: HyConGAT,
    loader: DataLoader,
    device: str,
    adjacency: torch.Tensor,
    target_node_indices: torch.Tensor,
    target_log1p: bool,
    target_names: List[str],
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, np.ndarray]:
    model.eval()
    true_batches: List[np.ndarray] = []
    prediction_batches: List[np.ndarray] = []

    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device), adjacency, target_node_indices)
            true_batches.append(targets.numpy())
            prediction_batches.append(predictions.cpu().numpy())

    y_true = np.concatenate(true_batches, axis=0)
    y_pred = np.concatenate(prediction_batches, axis=0)

    if target_log1p:
        y_true = inverse_safe_log1p(y_true)
        y_pred = inverse_safe_log1p(y_pred)

    metrics = {"global": compute_metrics(y_true.reshape(-1), y_pred.reshape(-1))}
    for target_index, target_name in enumerate(target_names):
        metrics[target_name] = compute_metrics(y_true[:, target_index], y_pred[:, target_index])
    return metrics, y_true, y_pred


def train_model(
    model: HyConGAT,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    cfg: Config,
    device: str,
    adjacency: torch.Tensor,
    target_node_indices: torch.Tensor,
    checkpoint_path: str,
) -> List[Dict[str, float]]:
    best_valid_loss = float("inf")
    remaining_patience = cfg.patience
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs, adjacency, target_node_indices), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                valid_loss = criterion(model(inputs, adjacency, target_node_indices), targets)
                valid_losses.append(float(valid_loss.item()))

        train_loss = float(np.mean(train_losses))
        valid_loss = float(np.mean(valid_losses))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "lr": learning_rate,
            }
        )
        scheduler.step(valid_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train={train_loss:.5f} | valid={valid_loss:.5f} | lr={learning_rate:.2e}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            remaining_patience = cfg.patience
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "valid_loss": valid_loss}, checkpoint_path)
        else:
            remaining_patience -= 1
            if remaining_patience <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    return history


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_checkpoint(path: str, device: str) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def metrics_to_frame(split_name: str, metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for node_name, values in metrics.items():
        rows.append({"model": MODEL_NAME, "split": split_name, "node": node_name, **values})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    cfg.device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    tables_dir = os.path.join(cfg.out_dir, "tables")
    logs_dir = os.path.join(cfg.out_dir, "logs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Training {MODEL_NAME}")
    print(f"Device: {cfg.device}")

    nodes = pd.read_csv(cfg.node_csv)
    edges = pd.read_csv(cfg.edge_csv)
    daily = pd.read_csv(cfg.daily_csv)
    validate_columns(nodes, ["node_id", "node_type"], "node_csv")
    validate_columns(edges, ["from_id", "to_id", "weight_alpha"], "edge_csv")

    node_ids = nodes["node_id"].tolist()
    target_nodes = nodes.loc[nodes["node_type"] == "O", "node_id"].tolist()
    if not target_nodes:
        raise ValueError("No target nodes found. Expected node_type == 'O' in node_csv.")

    static_features = build_static_features(nodes, node_ids)
    dynamic_tensor, target_values, dates, dynamic_feature_names = build_dynamic_tensor(daily, node_ids, target_nodes)
    inputs, raw_targets, target_dates = build_sequence_samples(
        dynamic_tensor,
        target_values,
        static_features,
        dates,
        cfg.seq_len,
        cfg.pred_horizon,
    )
    train_idx, valid_idx, test_idx = split_by_target_date(target_dates, cfg)
    standardized_inputs, feature_mean, feature_std = standardize_by_training_set(inputs, train_idx)
    targets = safe_log1p(raw_targets) if cfg.target_log1p else raw_targets.copy()

    train_dataset = SequenceDataset(standardized_inputs, targets, train_idx)
    valid_dataset = SequenceDataset(standardized_inputs, targets, valid_idx)
    test_dataset = SequenceDataset(standardized_inputs, targets, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    adjacency_np = build_adjacency(node_ids, edges, cfg.add_reverse_edges, cfg.add_self_loops)
    adjacency_norm_np = normalize_adjacency(adjacency_np)
    adjacency = torch.tensor(adjacency_np, dtype=torch.float32, device=cfg.device)
    target_node_indices_np = np.array([node_ids.index(node) for node in target_nodes], dtype=np.int64)
    target_node_indices = torch.tensor(target_node_indices_np, dtype=torch.long, device=cfg.device)

    target_weights = parse_target_weights(cfg.target_loss_weights, target_nodes)
    node_weights = torch.tensor([target_weights[node] for node in target_nodes], dtype=torch.float32, device=cfg.device)

    model = HyConGAT(
        in_dim=standardized_inputs.shape[-1],
        graph_out_dim=cfg.graph_out_dim,
        hidden_dim=cfg.hidden_dim,
        target_count=len(target_nodes),
        graph_layers=cfg.graph_layers,
        gat_heads=cfg.gat_heads,
        gru_layers=cfg.gru_layers,
        dropout=cfg.dropout,
        use_residual_graph=cfg.use_residual_graph,
        add_target_id_embedding=cfg.add_target_id_embedding and len(target_nodes) > 1,
    ).to(cfg.device)

    criterion = FrequencyWeightedHuberLoss(
        node_weights=node_weights,
        training_targets=targets[train_idx],
        extreme_weight_lambda=cfg.loss_extreme_weight_lambda,
        delta=cfg.huber_delta,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    checkpoint_path = os.path.join(logs_dir, "best_hycongat.pt")
    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=cfg.device,
        adjacency=adjacency,
        target_node_indices=target_node_indices,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = load_checkpoint(checkpoint_path, cfg.device)
    model.load_state_dict(checkpoint["model_state"])

    valid_metrics, _, _ = evaluate(
        model,
        valid_loader,
        cfg.device,
        adjacency,
        target_node_indices,
        cfg.target_log1p,
        target_nodes,
    )
    test_metrics, y_true_test, y_pred_test = evaluate(
        model,
        test_loader,
        cfg.device,
        adjacency,
        target_node_indices,
        cfg.target_log1p,
        target_nodes,
    )

    history_path = os.path.join(tables_dir, "training_history.csv")
    metrics_path = os.path.join(tables_dir, "metrics.csv")
    predictions_path = os.path.join(tables_dir, "test_predictions.csv")
    artifacts_path = os.path.join(logs_dir, "training_artifacts.npz")
    metadata_path = os.path.join(logs_dir, "metadata.json")

    pd.DataFrame(history).to_csv(history_path, index=False)
    pd.concat(
        [metrics_to_frame("valid", valid_metrics), metrics_to_frame("test", test_metrics)],
        ignore_index=True,
    ).to_csv(metrics_path, index=False)

    test_dates = pd.to_datetime(target_dates[test_idx])
    prediction_rows = []
    for row_index, date in enumerate(test_dates):
        row = {"date": str(date.date())}
        for target_index, target_name in enumerate(target_nodes):
            row[f"{target_name}_true"] = float(y_true_test[row_index, target_index])
            row[f"{target_name}_pred"] = float(y_pred_test[row_index, target_index])
            row[f"{target_name}_residual"] = float(y_pred_test[row_index, target_index] - y_true_test[row_index, target_index])
        prediction_rows.append(row)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    feature_names = dynamic_feature_names + STATIC_FEATURE_COLUMNS
    np.savez_compressed(
        artifacts_path,
        train_idx=train_idx.astype(np.int64),
        valid_idx=valid_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        target_dates=np.array(pd.to_datetime(target_dates).astype(str), dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        node_ids=np.array(node_ids, dtype=object),
        target_nodes=np.array(target_nodes, dtype=object),
        adjacency=adjacency_np.astype(np.float32),
        adjacency_norm=adjacency_norm_np.astype(np.float32),
        target_node_indices=target_node_indices_np.astype(np.int64),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
    )

    metadata = {
        "model_name": MODEL_NAME,
        "config": asdict(cfg),
        "dynamic_feature_columns": dynamic_feature_names,
        "static_feature_columns": STATIC_FEATURE_COLUMNS,
        "target_nodes": target_nodes,
        "target_loss_weights": target_weights,
        "outputs": {
            "checkpoint": checkpoint_path,
            "history_csv": history_path,
            "metrics_csv": metrics_path,
            "test_predictions_csv": predictions_path,
            "training_artifacts_npz": artifacts_path,
        },
    }
    save_json(metadata_path, metadata)

    print("\nTest metrics")
    for node_name, values in test_metrics.items():
        formatted = ", ".join(f"{metric}={score:.4f}" for metric, score in values.items())
        print(f"  {node_name}: {formatted}")

    print("\nSaved outputs")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  predictions: {predictions_path}")
    print(f"  metadata: {metadata_path}")


if __name__ == "__main__":
    main()
