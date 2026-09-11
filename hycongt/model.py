from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import DifferentiableSRTOBalance


class DirectedGATLayer(nn.Module):
    """Multi-head graph attention with information flow from source to destination."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads.")
        self.heads = int(heads)
        self.head_dim = out_dim // heads
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attention)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, _ = x.shape
        h = self.linear(x).view(batch, n_nodes, self.heads, self.head_dim)

        h_source = h.unsqueeze(2).expand(batch, n_nodes, n_nodes, self.heads, self.head_dim)
        h_target = h.unsqueeze(1).expand(batch, n_nodes, n_nodes, self.heads, self.head_dim)
        pair = torch.cat([h_source, h_target], dim=-1)
        score = self.leaky_relu((pair * self.attention).sum(dim=-1))

        mask = adjacency.bool().unsqueeze(0).unsqueeze(-1)
        score = score.masked_fill(~mask, -torch.inf)
        weight = torch.softmax(score, dim=1)
        weight = self.dropout(weight)
        out = (weight.unsqueeze(-1) * h_source).sum(dim=1)
        return out.reshape(batch, n_nodes, -1) + self.out_bias


class HyConGT(nn.Module):
    """Three directed GAT layers, one GRU layer, and differentiable SRTO balance."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        n_nodes: int,
        graph_dim: int = 64,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
        balance_iters: int = 3,
        delta_alpha_max: float = 0.25,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.dynamic_dim = int(dynamic_dim)
        self.static_dim = int(static_dim)
        self.delta_alpha_max = float(delta_alpha_max)

        self.input_projection = nn.Linear(dynamic_dim + static_dim, graph_dim)
        self.gat1 = DirectedGATLayer(graph_dim, graph_dim, gat_heads, dropout)
        self.gat2 = DirectedGATLayer(graph_dim, graph_dim, gat_heads, dropout)
        self.gat3 = DirectedGATLayer(graph_dim, graph_dim, gat_heads, dropout)
        self.norm1 = nn.LayerNorm(graph_dim)
        self.norm2 = nn.LayerNorm(graph_dim)
        self.norm3 = nn.LayerNorm(graph_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(graph_dim, hidden_dim, num_layers=1, batch_first=True)

        self.node_head = nn.Linear(hidden_dim, 12)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.initial_storage_logit = nn.Parameter(torch.full((n_nodes,), -0.85))
        self.balance = DifferentiableSRTOBalance(balance_iters, delta_alpha_max)
        self.register_buffer("adjacency", torch.eye(n_nodes, dtype=torch.bool))

        nn.init.zeros_(self.node_head.weight)
        nn.init.zeros_(self.node_head.bias)
        # Initialization mapped to the Table S6 output order. These values preserve
        # the stable starting point used by the research implementation.
        with torch.no_grad():
            self.node_head.bias[7] = -1.2   # retention release fraction r
            self.node_head.bias[8] = 1.4    # treatment fraction p
            self.node_head.bias[9] = 2.2    # hydraulic return fraction eta
            self.node_head.bias[11] = -1.0  # latent operational-water term

    def set_graph(self, edge_src: torch.Tensor, edge_dst: torch.Tensor) -> None:
        adjacency = torch.eye(self.n_nodes, device=edge_src.device, dtype=torch.bool)
        adjacency[edge_src, edge_dst] = True
        self.adjacency = adjacency

    def encode_step(self, node_features: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_projection(node_features))
        h = self.norm1(h + self.dropout(F.elu(self.gat1(h, self.adjacency))))
        h = self.norm2(h + self.dropout(F.elu(self.gat2(h, self.adjacency))))
        h = self.norm3(h + self.dropout(F.elu(self.gat3(h, self.adjacency))))
        return h

    def params_from_hidden(
        self,
        hidden: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map GRU states to the bounded outputs listed in Supplementary Table S6."""

        raw = self.node_head(hidden)
        params = {
            "k_c": 0.5 + torch.sigmoid(raw[..., 0]),
            "k_snow": 0.5 + torch.sigmoid(raw[..., 1]),
            "k_gw": 3.0 * torch.sigmoid(raw[..., 2]),
            "k_mine": 0.5 + torch.sigmoid(raw[..., 3]),
            "k_intake": 2.0 * torch.sigmoid(raw[..., 4]),
            "k_rec": 3.0 * torch.sigmoid(raw[..., 5]),
            "k_e": 0.5 + torch.sigmoid(raw[..., 6]),
            "r": torch.sigmoid(raw[..., 7]),
            "p": torch.sigmoid(raw[..., 8]),
            "eta": torch.sigmoid(raw[..., 9]),
            "q_intake_latent": F.softplus(raw[..., 10]).clamp_max(8.0e4),
            "q_op_latent": 8.0e4 * torch.sigmoid(raw[..., 11]),
        }

        h_source = hidden[:, edge_src]
        h_target = hidden[:, edge_dst]
        edge_raw = self.edge_head(torch.cat([h_source, h_target], dim=-1)).squeeze(-1)
        params["delta_alpha"] = self.delta_alpha_max * torch.tanh(edge_raw)
        return params

    def initial_storage(self, vmax: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        fraction = torch.sigmoid(self.initial_storage_logit).unsqueeze(0)
        return fraction.expand(batch_size, -1) * vmax.unsqueeze(0)
