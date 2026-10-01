from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics import DifferentiableSRTOBalance


class DirectedGATLayer(nn.Module):
    """Multi-head GAT on unidirectional adjacency (source -> destination only)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if heads < 1 or out_dim % heads:
            raise ValueError('GAT output dimension must be divisible by positive head count')
        self.heads = heads
        self.dh = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.zeros(heads, 2 * self.dh))
        nn.init.xavier_uniform_(self.attn)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        self.out_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.heads, self.dh)
        h_i = h.unsqueeze(2).expand(B, N, N, self.heads, self.dh)
        h_j = h.unsqueeze(1).expand(B, N, N, self.heads, self.dh)
        cat = torch.cat([h_i, h_j], dim=-1)
        e = self.leaky((cat * self.attn).sum(-1))
        mask = (adj > 0).float().unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask.squeeze(-1).unsqueeze(-1) <= 0, -1e9)
        alpha = torch.softmax(e, dim=1)
        alpha = self.dropout(alpha)
        out = (alpha.unsqueeze(-1) * h_i).sum(dim=1)
        out = out.reshape(B, N, self.heads * self.dh) + self.out_bias
        return out


class PEGATGRU(nn.Module):
    """3-layer directed GAT + 1-layer GRU -> SRTO parameters -> water-balance Q only."""

    def __init__(
        self,
        dyn_dim: int,
        static_dim: int,
        n_nodes: int,
        n_edges: int,
        graph_dim: int = 64,
        hidden_dim: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
        balance_iters: int = 3,
        delta_alpha_max: float = 0.25,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.delta_alpha_max = float(delta_alpha_max)
        in_dim = dyn_dim + static_dim
        self.input_proj = nn.Linear(in_dim, graph_dim)
        self.gat1 = DirectedGATLayer(graph_dim, graph_dim, heads=gat_heads, dropout=dropout)
        self.gat2 = DirectedGATLayer(graph_dim, graph_dim, heads=gat_heads, dropout=dropout)
        self.gat3 = DirectedGATLayer(graph_dim, graph_dim, heads=gat_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(graph_dim)
        self.norm2 = nn.LayerNorm(graph_dim)
        self.norm3 = nn.LayerNorm(graph_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(graph_dim, hidden_dim, num_layers=1, batch_first=True)

        # k_c,k_snow,k_mine,k_intake,k_e,r,p,eta,q_intake_free,k_rec,k_gw,q_ext
        self.node_head = nn.Linear(hidden_dim, 12)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.rho_logit = nn.Parameter(torch.full((n_nodes,), -0.85))
        # Trained causal observation gate (gauged nodes only): Q = (1-g) Q_srto + g Q_lag1_obs
        self.obs_gate = nn.Linear(hidden_dim, 1)
        self.balance = DifferentiableSRTOBalance(
            n_iters=balance_iters, delta_alpha_max=self.delta_alpha_max
        )
        nn.init.zeros_(self.node_head.weight)
        nn.init.zeros_(self.node_head.bias)
        self.node_head.bias.data[5] = -1.2
        self.node_head.bias.data[6] = 1.4
        self.node_head.bias.data[7] = 2.2
        self.node_head.bias.data[9] = 0.0
        self.node_head.bias.data[10] = 0.0
        self.node_head.bias.data[11] = -1.0
        nn.init.zeros_(self.obs_gate.weight)
        nn.init.constant_(self.obs_gate.bias, -0.5) # mild preference for physics at start
        self.register_buffer("adj", torch.eye(n_nodes))

    def set_graph(self, edge_src: torch.Tensor, edge_dst: torch.Tensor) -> None:
        N = self.n_nodes
        adj = torch.eye(N, device=edge_src.device)
        for s, d in zip(edge_src.tolist(), edge_dst.tolist()):
            adj[s, d] = 1.0
        self.adj = adj

    def encode_step(self, x_t: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x_t))
        h = self.norm1(h + self.dropout(F.elu(self.gat1(h, self.adj))))
        h = self.norm2(h + self.dropout(F.elu(self.gat2(h, self.adj))))
        h = self.norm3(h + self.dropout(F.elu(self.gat3(h, self.adj))))
        return h

    def params_from_hidden(
        self,
        h_nodes: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raw = self.node_head(h_nodes)
        k_c = 0.5 + torch.sigmoid(raw[..., 0])
        k_snow = 0.5 + torch.sigmoid(raw[..., 1])
        k_mine = 0.5 + torch.sigmoid(raw[..., 2])
        k_intake = 2.0 * torch.sigmoid(raw[..., 3])
        k_e = 0.5 + torch.sigmoid(raw[..., 4])
        r = torch.sigmoid(raw[..., 5])
        p = torch.sigmoid(raw[..., 6])
        eta = torch.sigmoid(raw[..., 7])
        q_intake_free = raw[..., 8]
        k_rec = 3.0 * torch.sigmoid(raw[..., 9])
        k_gw = 3.0 * torch.sigmoid(raw[..., 10])
        q_ext = raw[..., 11]
        hs = h_nodes[:, edge_src]
        hd = h_nodes[:, edge_dst]
        # bounded edge correction δα
        delta_alpha = self.delta_alpha_max * torch.tanh(
            self.edge_head(torch.cat([hs, hd], dim=-1)).squeeze(-1)
        )
        return {
            "k_c": k_c,
            "k_snow": k_snow,
            "k_mine": k_mine,
            "k_intake": k_intake,
            "k_e": k_e,
            "r": r,
            "p": p,
            "eta": eta,
            "q_intake_free": q_intake_free,
            "k_rec": k_rec,
            "k_gw": k_gw,
            "q_ext": q_ext,
            "delta_alpha": delta_alpha,
        }

    def initial_storage(self, vmax: torch.Tensor, batch: int) -> torch.Tensor:
        rho = torch.sigmoid(self.rho_logit).unsqueeze(0).expand(batch, -1)
        return rho * vmax.unsqueeze(0)
