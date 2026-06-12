#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x_nodes: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("ij,bjd->bid", adj_norm, x_nodes)
        return self.linear(h)


class TemporalConvBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )
        self.skip = nn.Identity() if in_dim == out_dim else nn.Conv1d(in_dim, out_dim, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(out_dim)
        self.norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.pad(x, (self.left_padding, 0))
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.pad(x, (self.left_padding, 0))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)



class GCNTemporalEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        node_base_dim: int,
        projection_dim: int,
        gcn_hidden_dim: int,
        temporal_hidden_dim: int,
        edges: list[tuple[int, int]],
    ) -> None:
        super().__init__()

        self.node_dim = node_dim
        self.node_base_dim = node_base_dim
        self.num_nodes = num_nodes
        self.num_joints = num_nodes - 1

        adj = self._build_adj(num_nodes, edges)
        self.register_buffer("adj_norm", self._normalize_adj(adj))
        self.register_buffer("joint_permutation", torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=torch.long))

        self.node_base_projection = nn.Linear(node_base_dim, projection_dim)
        self.node_joint_projection = nn.Linear(node_dim, projection_dim)

        self.gcn1 = GCNLayer(projection_dim, gcn_hidden_dim)
        self.gcn2 = GCNLayer(gcn_hidden_dim, gcn_hidden_dim)
        # self.pre_temporal_mlp = nn.Sequential(
        #     nn.Linear(node_base_dim + self.num_joints * self.node_dim + 4, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, gcn_hidden_dim),
        # )

        # self.temporal_encoder = nn.GRU(
        #     input_size=gcn_hidden_dim,
        #     hidden_size=temporal_hidden_dim,
        #     batch_first=True,
        # )
        # self.temporal_conv = nn.Sequential(
        #     TemporalConvBlock(gcn_hidden_dim, temporal_hidden_dim, kernel_size=3, dilation=1),
        #     TemporalConvBlock(temporal_hidden_dim, temporal_hidden_dim, kernel_size=3, dilation=2),
        #     TemporalConvBlock(temporal_hidden_dim, temporal_hidden_dim, kernel_size=3, dilation=4),
        # )


    @staticmethod
    def _build_adj(num_nodes: int, edges: list[tuple[int, int]]) -> torch.Tensor:
        adj = torch.zeros(num_nodes, num_nodes)
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj = adj + torch.eye(num_nodes)
        return adj

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        d_inv_sqrt = torch.diag(degree_inv_sqrt)
        return d_inv_sqrt @ adj @ d_inv_sqrt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, history_length, feature_dim = x.shape
        expected_dim = self.node_base_dim + self.num_joints * self.node_dim + 4
        if feature_dim != expected_dim:
            raise ValueError(f"Expected input feature_dim={expected_dim}, got {feature_dim}")

        # hid = self.pre_temporal_mlp(x)
        
        # GCN path kept here for comparison against the simpler per-timestep MLP encoder.
        x_base = x[:, :, 24:33]
        pos, vel, a_prev = x[:, :, :12], x[:, :, 12:24], x[:, :, 33:45]
        #['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 
        # 'RR_hip_joint', 'FL_thigh_joint', 'FR_thigh_joint', 
        # 'RL_thigh_joint', 'RR_thigh_joint', 'FL_calf_joint', 
        # 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint']
        x_joints = torch.stack(
            [
                pos[:, :, self.joint_permutation],
                vel[:, :, self.joint_permutation],
                a_prev[:, :, self.joint_permutation],
            ],
            dim=-1,
        )

        x_base = self.node_base_projection(x_base).unsqueeze(2)
        x_joints = self.node_joint_projection(x_joints)
        
        x_nodes = torch.cat([x_joints, x_base], dim=2)
        # breakpoint()
        # Merge batch and time before message passing to avoid expensive 4D contractions.
        # x_nodes = x_nodes.reshape(batch_size * history_length, self.num_nodes, -1)
        hid = F.relu(self.gcn1(x_nodes, self.adj_norm))
        hid = F.relu(self.gcn2(hid, self.adj_norm))
        hid = hid.mean(dim=2).reshape(hid.shape[0], -1)
        # breakpoint()
        # hid = hid.view(batch_size, history_length, -1)

        # GRU baseline kept here for reference.
        # out, _ = self.temporal_encoder(hid)
        # last = out.mean(1) 
        # last = hid.mean(1)
        # return last
        # hid = hid.transpose(1, 2)
        # hid = self.temporal_conv(hid)
        # return hid.mean(-1)
        # # last = hid[:, :, -1]
        # last = hid.mean(-1)
        # # breakpoint()
        return hid


class HistoryRegressor(nn.Module):
    def __init__(
        self,
        history_length: int,
        target_dim: int,
        projection_dim: int,
        gcn_hidden_dim: int,
        temporal_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.history_length = history_length
        self.target_dim = target_dim

        edges = [
            (0, 1), (1, 2),
            (3, 4), (4, 5),
            (6, 7), (7, 8),
            (9, 10), (10, 11),
            (12, 0),
            (12, 3),
            (12, 6),
            (12, 9),
        ]

        self.encoder = GCNTemporalEncoder(
            num_nodes=13,
            node_dim=3,
            node_base_dim=6,
            projection_dim=projection_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            temporal_hidden_dim=temporal_hidden_dim,
            edges=edges,
        )

        self.decoder = nn.Sequential(
            nn.LayerNorm(temporal_hidden_dim),
            nn.Linear(temporal_hidden_dim, temporal_hidden_dim),
            nn.ReLU(),
            nn.Linear(temporal_hidden_dim, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class SlidingWindowAccelerationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        data_gnn: torch.Tensor,
        data_label: torch.Tensor,
        done_flags: torch.Tensor,
        history_length: int,
    ) -> None:
        if data_gnn.ndim != 3 or data_label.ndim != 3:
            raise ValueError("Expected data_gnn and data_label to both have shape [T, E, D].")
        if done_flags.ndim != 2:
            raise ValueError("Expected done_flags to have shape [T, E].")
        if data_gnn.shape[:2] != done_flags.shape:
            raise ValueError("data_gnn and done_flags must share the same [T, E] dimensions.")
        if data_label.shape[0] != data_gnn.shape[0] - 2 or data_label.shape[1] != data_gnn.shape[1]:
            raise ValueError("data_label must have shape [T - 2, E, target_dim].")
        if history_length < 2:
            raise ValueError("history_length must be at least 2.")

        self.data_gnn = data_gnn.contiguous()
        self.data_label = data_label.contiguous()
        self.done_flags = done_flags.to(dtype=torch.bool).contiguous()
        self.history_length = history_length
        self.indices = self._build_indices()

        if not self.indices:
            raise ValueError("No valid windows were found. Try reducing history_length or checking the dataset.")
        self.index_tensor = torch.tensor(self.indices, dtype=torch.long)

    def _build_indices(self) -> list[tuple[int, int]]:
        time_steps, num_envs, _ = self.data_gnn.shape
        valid_indices: list[tuple[int, int]] = []

        for env_idx in range(num_envs):
            done_cumsum = torch.zeros(time_steps + 1, dtype=torch.int32)
            done_cumsum[1:] = self.done_flags[:, env_idx].to(torch.int32).cumsum(dim=0)

            for target_t in range(self.history_length - 1, time_steps - 1):
                # Window uses observations [target_t - H + 1, ..., target_t].
                # Acceleration label is centered at target_t and uses q(target_t - 1), q(target_t), q(target_t + 1).
                start_t = target_t - self.history_length + 1
                end_t = target_t + 1
                if done_cumsum[end_t + 1] - done_cumsum[start_t] != 0:
                    continue
                valid_indices.append((env_idx, target_t))

        return valid_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        env_idx, target_t = self.indices[index]
        start_t = target_t - self.history_length + 1

        window = self.data_gnn[start_t : target_t + 1, env_idx, :]
        target = self.data_label[target_t - 1, env_idx, :]
        return window, target


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_acceleration_labels(joint_pos: torch.Tensor, dt: float) -> torch.Tensor:
    if joint_pos.ndim != 3:
        raise ValueError("Expected joint_pos with shape [T, E, num_joints].")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    return (joint_pos[2:] + joint_pos[:-2] - 2.0 * joint_pos[1:-1]) / (dt * dt)


def load_history_tensors(
    data_path: Path,
    dt: float,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.load(data_path, allow_pickle=True).astype(np.float32)
    if limit is not None:
        data = data[:limit]

    if data.ndim != 3 or data.shape[-1] < 67:
        raise ValueError("Expected raw dataset with shape [T, E, 67].")

    data_tensor = torch.from_numpy(data.copy())
    data_gnn = data_tensor[:, :, :42].contiguous()
    joint_pos = data_tensor[:, :, :12].contiguous()
    done_flags = data_tensor[:, :, -1].contiguous()
    data_label = compute_acceleration_labels(joint_pos, dt=dt)
    return data_gnn, data_label, done_flags


def build_dataloaders(
    data_gnn: torch.Tensor,
    data_label: torch.Tensor,
    done_flags: torch.Tensor,
    history_length: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = SlidingWindowAccelerationDataset(
        data_gnn=data_gnn,
        data_label=data_label,
        done_flags=done_flags,
        history_length=history_length,
    )

    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    if min(train_len, val_len, test_len) <= 0:
        raise ValueError("Split produced an empty train/val/test set. Adjust train_ratio/val_ratio.")

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def compute_subset_target_stats(
    dataset: SlidingWindowAccelerationDataset,
    subset_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_index_tensor = torch.as_tensor(subset_indices, dtype=torch.long)
    index_pairs = dataset.index_tensor[sample_index_tensor]

    env_indices = index_pairs[:, 0]
    target_times = index_pairs[:, 1] - 1
    targets = dataset.data_label[target_times, env_indices, :]

    mean = targets.mean(dim=0)
    std = targets.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0
    total_squared_error = 0.0
    total_target_values = 0

    with torch.set_grad_enabled(is_train):
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            preds_norm = model(batch_inputs)
            targets_norm = (batch_targets - target_mean) / target_std
            loss = criterion(preds_norm, targets_norm)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            preds_raw = preds_norm * target_std + target_mean
            squared_error = torch.square(preds_raw - batch_targets).sum()

            batch_size = batch_inputs.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
            total_squared_error += squared_error.item()
            total_target_values += batch_targets.numel()

    avg_loss = total_loss / total_items
    rmse = (total_squared_error / total_target_values) ** 0.5
    return avg_loss, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GCN+GRU regressor on quad history data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/home/dung-admin/quadloco_ws/quadlocofault/quad_dynamic_dataset_20000.npy"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/dung-admin/quadloco_ws/quadlocofault/outputs/gnn_history_regressor.pt"),
    )
    parser.add_argument("--history-length", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.005, help="Physics timestep used when the dataset was logged.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset cap for debugging.")
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--gcn-hidden-dim", type=int, default=32)
    parser.add_argument("--temporal-hidden-dim", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_gnn, data_label, done_flags = load_history_tensors(args.data_path, dt=args.dt, limit=args.limit)
    history_length = args.history_length
    target_dim = data_label.shape[-1]

    print(f"Loaded raw data_gnn:   {tuple(data_gnn.shape)}")
    print(f"Loaded accel labels:   {tuple(data_label.shape)}")
    print(f"History length:        {history_length}")
    print(f"Physics dt:            {args.dt}")
    print(f"Device:                {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_gnn=data_gnn,
        data_label=data_label,
        done_flags=done_flags,
        history_length=history_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_mean, train_std = compute_subset_target_stats(train_loader.dataset.dataset, train_loader.dataset.indices)
    target_mean = train_mean.to(device)
    target_std = train_std.to(device)

    print(f"Train target mean abs: {train_mean.abs().mean().item():.4f}")
    print(f"Train target std mean: {train_std.mean().item():.4f}")

    model = HistoryRegressor(
        history_length=history_length,
        target_dim=target_dim,
        projection_dim=args.projection_dim,
        gcn_hidden_dim=args.gcn_hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_rmse = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            target_mean,
            target_std,
            optimizer=optimizer,
        )
        val_loss, val_rmse = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            target_mean,
            target_std,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmse = val_rmse
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss_norm={train_loss:.6f} "
            f"train_rmse={train_rmse:.6f} "
            f"val_loss_norm={val_loss:.6f} "
            f"val_rmse={val_rmse:.6f}"
        )

    model.load_state_dict(best_state_dict)
    test_loss, test_rmse = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        target_mean,
        target_std,
    )
    print(f"test_loss_norm={test_loss:.6f} test_rmse={test_rmse:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_rmse": best_val_rmse,
            "test_loss": test_loss,
            "test_rmse": test_rmse,
            "history_length": history_length,
            "target_dim": target_dim,
            "dt": args.dt,
            "target_mean": train_mean,
            "target_std": train_std,
            "args": vars(args),
        },
        args.output,
    )
    print(f"Saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
