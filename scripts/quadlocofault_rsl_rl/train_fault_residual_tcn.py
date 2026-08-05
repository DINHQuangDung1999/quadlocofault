"""Train FaultResidualTCN as an offline 13-class joint-fault classifier."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from models.equiv_gcn_actor import FaultResidualTCN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the exact FaultResidualTCN used by EquivGCNActor on collected "
            "history with 12 faulty-joint classes plus one healthy class."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Directory containing shard_*.pt files from run_collect_prop_data.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Training output directory. Defaults to <dataset_dir>/fault_tcn_runs/<timestamp>.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument(
        "--film_dim",
        type=int,
        default=16,
        help="Kept compatible with EquivGCNActor; the FiLM head is frozen here.",
    )
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument(
        "--min_fault_age_steps",
        type=int,
        default=0,
        help="Discard samples whose current oracle label is younger than this many steps.",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
        help="Keep every Nth selected sample within each shard.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )
    return parser.parse_args()


def load_pt(path: Path) -> dict[str, Any]:
    """Load a collector file across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass
class DatasetSplitIndex:
    shard_paths: list[Path]
    train_indices: list[torch.Tensor]
    validation_indices: list[torch.Tensor]
    train_positive_bits: int
    train_total_bits: int
    metadata: dict[str, Any]


def build_split_index(
    shard_paths: list[Path],
    validation_fraction: float,
    seed: int,
    min_fault_age_steps: int,
    sample_stride: int,
) -> DatasetSplitIndex:
    """Build an episode-level split without retaining history tensors in memory."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be between zero and one, got {validation_fraction}."
        )
    if min_fault_age_steps < 0:
        raise ValueError(
            f"min_fault_age_steps must be non-negative, got {min_fault_age_steps}."
        )
    if sample_stride <= 0:
        raise ValueError(f"sample_stride must be positive, got {sample_stride}.")

    episode_ids_by_shard: list[torch.Tensor] = []
    fault_age_by_shard: list[torch.Tensor] = []
    metadata: dict[str, Any] | None = None

    print(f"[INFO] Scanning {len(shard_paths)} dataset shards...")
    for shard_path in shard_paths:
        shard = load_pt(shard_path)
        required = {"history", "fault_target", "episode_id", "fault_age_steps"}
        missing = required.difference(shard)
        if missing:
            raise KeyError(f"{shard_path} is missing fields: {sorted(missing)}")
        if shard["history"].ndim != 3 or shard["history"].shape[-1] != 45:
            raise ValueError(
                f"{shard_path} has history shape {tuple(shard['history'].shape)}; "
                "expected [samples, history_length, 45]."
            )
        if shard["fault_target"].shape != (shard["history"].shape[0], 12):
            raise ValueError(
                f"{shard_path} has fault_target shape {tuple(shard['fault_target'].shape)}; "
                f"expected ({shard['history'].shape[0]}, 12)."
            )

        episode_ids_by_shard.append(shard["episode_id"].reshape(-1).to(torch.long).clone())
        fault_age_by_shard.append(
            shard["fault_age_steps"].reshape(-1).to(torch.long).clone()
        )
        if metadata is None:
            metadata = dict(shard.get("metadata", {}))
        del shard

    all_episode_ids = torch.unique(torch.cat(episode_ids_by_shard))
    if all_episode_ids.numel() < 2:
        raise ValueError("At least two distinct episodes are required for train/validation.")

    generator = torch.Generator().manual_seed(seed)
    shuffled_episode_ids = all_episode_ids[
        torch.randperm(all_episode_ids.numel(), generator=generator)
    ]
    num_validation_episodes = round(validation_fraction * all_episode_ids.numel())
    num_validation_episodes = min(
        max(num_validation_episodes, 1), all_episode_ids.numel() - 1
    )
    validation_episode_ids = shuffled_episode_ids[:num_validation_episodes]

    train_indices: list[torch.Tensor] = []
    validation_indices: list[torch.Tensor] = []
    train_positive_bits = 0
    train_total_bits = 0

    for shard_path, episode_ids, fault_age in zip(
        shard_paths, episode_ids_by_shard, fault_age_by_shard
    ):
        is_validation = torch.isin(episode_ids, validation_episode_ids)
        age_is_valid = fault_age >= min_fault_age_steps

        shard_train_indices = torch.nonzero(
            ~is_validation & age_is_valid, as_tuple=False
        ).reshape(-1)[::sample_stride]
        shard_validation_indices = torch.nonzero(
            is_validation & age_is_valid, as_tuple=False
        ).reshape(-1)[::sample_stride]
        train_indices.append(shard_train_indices)
        validation_indices.append(shard_validation_indices)

        if shard_train_indices.numel():
            shard = load_pt(shard_path)
            train_target = shard["fault_target"][shard_train_indices]
            train_positive_bits += int(train_target.sum().item())
            train_total_bits += train_target.numel()
            del shard, train_target

    if sum(indices.numel() for indices in train_indices) == 0:
        raise ValueError("The filters left the training split empty.")
    if sum(indices.numel() for indices in validation_indices) == 0:
        raise ValueError("The filters left the validation split empty.")
    if train_positive_bits == 0:
        raise ValueError("The training split has no positive fault labels.")

    return DatasetSplitIndex(
        shard_paths=shard_paths,
        train_indices=train_indices,
        validation_indices=validation_indices,
        train_positive_bits=train_positive_bits,
        train_total_bits=train_total_bits,
        metadata=metadata or {},
    )


class FaultHistoryDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy shard-backed dataset returning history and a 13-class target index."""

    def __init__(self, shard_paths: list[Path], indices_by_shard: list[torch.Tensor]) -> None:
        self.shard_paths: list[Path] = []
        self.indices_by_shard: list[torch.Tensor] = []
        self.shard_lengths: list[int] = []

        for shard_path, indices in zip(shard_paths, indices_by_shard):
            if indices.numel() == 0:
                continue
            self.shard_paths.append(shard_path)
            self.indices_by_shard.append(indices)
            self.shard_lengths.append(indices.numel())

        self.cumulative_ends: list[int] = []
        cumulative_length = 0
        for shard_length in self.shard_lengths:
            cumulative_length += shard_length
            self.cumulative_ends.append(cumulative_length)

        self._cached_shard_index: int | None = None
        self._cached_history: torch.Tensor | None = None
        self._cached_target: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.cumulative_ends[-1] if self.cumulative_ends else 0

    def _load_shard(self, shard_index: int) -> None:
        if self._cached_shard_index == shard_index:
            return
        shard = load_pt(self.shard_paths[shard_index])
        self._cached_history = shard["history"]
        self._cached_target = shard["fault_target"]
        self._cached_shard_index = shard_index

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        shard_index = bisect.bisect_right(self.cumulative_ends, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_ends[shard_index - 1]
        selected_offset = index - shard_start
        source_index = int(self.indices_by_shard[shard_index][selected_offset])

        self._load_shard(shard_index)
        assert self._cached_history is not None
        assert self._cached_target is not None
        fault_target = self._cached_target[source_index].bool()
        if int(fault_target.sum()) > 1:
            raise ValueError(
                "The 13-class objective requires at most one faulty joint per sample."
            )
        healthy_class = fault_target.numel()
        fault_class = (
            fault_target.to(torch.float32).argmax()
            if fault_target.any()
            else torch.tensor(healthy_class, dtype=torch.long)
        )
        return self._cached_history[source_index].to(torch.float32), fault_class.to(torch.long)


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle samples within shards while loading each shard roughly once per epoch."""

    def __init__(
        self,
        dataset: FaultHistoryDataset,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.shard_starts = [0]
        for length in dataset.shard_lengths[:-1]:
            self.shard_starts.append(self.shard_starts[-1] + length)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        shard_order = torch.randperm(len(self.dataset.shard_lengths), generator=generator)

        for shard_index_tensor in shard_order:
            shard_index = int(shard_index_tensor)
            shard_length = self.dataset.shard_lengths[shard_index]
            shard_start = self.shard_starts[shard_index]
            sample_order = torch.randperm(shard_length, generator=generator)
            for start in range(0, shard_length, self.batch_size):
                batch_offsets = sample_order[start : start + self.batch_size]
                if self.drop_last and batch_offsets.numel() < self.batch_size:
                    continue
                yield (batch_offsets + shard_start).tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return sum(length // self.batch_size for length in self.dataset.shard_lengths)
        return sum(
            (length + self.batch_size - 1) // self.batch_size
            for length in self.dataset.shard_lengths
        )


class MulticlassFaultMetrics:
    """Accumulate 13-class fault metrics without storing predictions."""

    def __init__(self, healthy_class: int = 12) -> None:
        self.healthy_class = healthy_class
        self.loss_sum = 0.0
        self.num_samples = 0
        self.confidence_sum = 0.0
        self.fault_probability_sum = 0.0
        self.faulty_samples = 0
        self.predicted_faulty_samples = 0
        self.detection_true_positive = 0
        self.detection_false_positive = 0
        self.detection_false_negative = 0
        self.localization_correct = 0
        self.classification_correct = 0

    def update(
        self, loss: torch.Tensor, logits: torch.Tensor, target: torch.Tensor
    ) -> None:
        probability = torch.softmax(logits, dim=-1)
        predicted_class = probability.argmax(dim=-1)
        batch_size = target.shape[0]

        self.loss_sum += float(loss.detach()) * batch_size
        self.num_samples += batch_size
        self.confidence_sum += float(probability.amax(dim=-1).sum())
        self.fault_probability_sum += float(
            (1.0 - probability[:, self.healthy_class]).sum()
        )

        has_fault = target != self.healthy_class
        predicts_fault = predicted_class != self.healthy_class
        self.faulty_samples += int(has_fault.sum())
        self.predicted_faulty_samples += int(predicts_fault.sum())
        self.detection_true_positive += int((has_fault & predicts_fault).sum())
        self.detection_false_positive += int((~has_fault & predicts_fault).sum())
        self.detection_false_negative += int((has_fault & ~predicts_fault).sum())

        self.localization_correct += int(
            ((predicted_class == target) & has_fault).sum()
        )
        self.classification_correct += int((predicted_class == target).sum())

    @staticmethod
    def _divide(numerator: int | float, denominator: int | float) -> float:
        return float(numerator) / max(float(denominator), 1.0)

    def compute(self) -> dict[str, float]:
        return {
            "loss": self._divide(self.loss_sum, self.num_samples),
            "mean_confidence": self._divide(
                self.confidence_sum, self.num_samples
            ),
            "mean_fault_probability": self._divide(
                self.fault_probability_sum, self.num_samples
            ),
            "faulty_sample_rate": self._divide(
                self.faulty_samples, self.num_samples
            ),
            "predicted_faulty_sample_rate": self._divide(
                self.predicted_faulty_samples, self.num_samples
            ),
            "detection_precision": self._divide(
                self.detection_true_positive,
                self.detection_true_positive + self.detection_false_positive,
            ),
            "detection_recall": self._divide(
                self.detection_true_positive,
                self.detection_true_positive + self.detection_false_negative,
            ),
            "localization_accuracy": self._divide(
                self.localization_correct, self.faulty_samples
            ),
            "classification_accuracy": self._divide(
                self.classification_correct, self.num_samples
            ),
        }


def run_epoch(
    model: FaultResidualTCN,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    gradient_clip: float,
    use_amp: bool,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    metrics = MulticlassFaultMetrics()

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for history, fault_class in loader:
            history = history.to(device=device, non_blocking=True)
            fault_class = fault_class.to(
                device=device, dtype=torch.long, non_blocking=True
            )

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                fault_logits, _, _ = model(history)
                loss = criterion(fault_logits, fault_class)

            if is_training:
                assert optimizer is not None
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

            metrics.update(loss, fault_logits.detach(), fault_class)

    return metrics.compute()


def save_checkpoint(
    path: Path,
    model: FaultResidualTCN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    dataset_metadata: dict[str, Any],
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    classifier_state_dict = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith("film_head.")
    }
    checkpoint = {
        "format_version": 2,
        "epoch": epoch,
        "model_class": "FaultResidualTCN",
        "model_kwargs": {
            "hidden_dim": args.hidden_dim,
            "film_dim": args.film_dim,
            "num_fault_classes": 13,
        },
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": classifier_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "classification_objective": "cross_entropy",
        "class_convention": "0..11=faulty joint, 12=healthy",
        "dataset_metadata": dataset_metadata,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "args": vars(args),
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def format_metrics(metrics: dict[str, float]) -> str:
    return (
        f"loss={metrics['loss']:.4f}, "
        f"class_acc={metrics['classification_accuracy']:.3f}, "
        f"localization={metrics['localization_accuracy']:.3f}, "
        f"detect_recall={metrics['detection_recall']:.3f}, "
        f"detect_precision={metrics['detection_precision']:.3f}, "
        f"confidence={metrics['mean_confidence']:.3f}"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}.")
    if args.gradient_clip <= 0.0:
        raise ValueError(f"gradient_clip must be positive, got {args.gradient_clip}.")

    shard_paths = sorted(args.dataset_dir.expanduser().resolve().glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {args.dataset_dir}.")

    split_index = build_split_index(
        shard_paths=shard_paths,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        min_fault_age_steps=args.min_fault_age_steps,
        sample_stride=args.sample_stride,
    )
    train_dataset = FaultHistoryDataset(
        split_index.shard_paths, split_index.train_indices
    )
    validation_dataset = FaultHistoryDataset(
        split_index.shard_paths, split_index.validation_indices
    )

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA was requested but is unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = requested_device
    use_amp = args.amp and device.type == "cuda"

    train_sampler = ShardBatchSampler(
        dataset=train_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = FaultResidualTCN(
        hidden_dim=args.hidden_dim,
        film_dim=args.film_dim,
        num_fault_classes=13,
    ).to(device)
    model.film_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.dataset_dir.expanduser().resolve() / "fault_tcn_runs" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose another --output_dir."
        )

    config_to_save = vars(args).copy()
    config_to_save["dataset_dir"] = str(args.dataset_dir)
    config_to_save["output_dir"] = str(output_dir)
    config_to_save["classification_objective"] = "cross_entropy"
    config_to_save["num_fault_classes"] = 13
    with (output_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config_to_save, config_file, indent=2)

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    target_rate = split_index.train_positive_bits / (
        split_index.train_total_bits / 12
    )
    print(f"[INFO] Device: {device}")
    print(
        f"[INFO] Samples: train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}"
    )
    print(
        f"[INFO] Training faulty-sample rate={target_rate:.6f}; "
        "objective=13-class cross-entropy"
    )

    best_validation_loss = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                gradient_clip=args.gradient_clip,
                use_amp=use_amp,
            )
            validation_metrics = run_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
                scaler=None,
                gradient_clip=args.gradient_clip,
                use_amp=use_amp,
            )
            scheduler.step(validation_metrics["loss"])

            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train: {format_metrics(train_metrics)} | "
                f"val: {format_metrics(validation_metrics)}"
            )

            if writer is not None:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch)
                for name, value in validation_metrics.items():
                    writer.add_scalar(f"validation/{name}", value, epoch)
                writer.add_scalar(
                    "train/learning_rate", optimizer.param_groups[0]["lr"], epoch
                )

            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                epoch,
                args,
                split_index.metadata,
                train_metrics,
                validation_metrics,
            )
            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    args,
                    split_index.metadata,
                    train_metrics,
                    validation_metrics,
                )
    finally:
        if writer is not None:
            writer.close()

    print(f"[INFO] Best validation loss: {best_validation_loss:.6f}")
    print(f"[INFO] Checkpoints saved in: {output_dir}")
    print(
        "[INFO] The checkpoint uses a 13-output fault head. Instantiate "
        "FaultResidualTCN with the saved model_kwargs before loading it."
    )


if __name__ == "__main__":
    main()
