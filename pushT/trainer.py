"""Training utilities and the Trainer for Push-T policies."""

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import TrainConfig
from data import PushTChunkDataset
from model import MSEPolicy


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to CPU if CUDA is unusable."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def build_dataloader(
    dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool = False
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW: Adam with decoupled weight decay (cfg.weight_decay=0 makes it plain Adam)."""
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )


def make_run_dir(cfg: TrainConfig) -> Path:
    """Create <output_dir>/<run_name>/ and return it. Raises if it already exists."""
    run_dir = cfg.output_dir / cfg.run_name
    run_dir.mkdir(parents=True)
    return run_dir


class Trainer:
    """Runs the optimisation loop, validation, logging and checkpointing."""

    def __init__(
        self,
        cfg: TrainConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        ckpt_extra: dict | None = None,
    ) -> None:
        self.cfg = cfg
        # Saved in every checkpoint, e.g. normalisation stats needed at evaluation time.
        self.ckpt_extra = ckpt_extra or {}
        self.device = resolve_device(cfg.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = build_optimizer(self.model, cfg)
        self.run_dir = make_run_dir(cfg)
        self.step = 0
        self.history: list[dict[str, float]] = []

    def _train_step(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """One optimisation step on a batch; return the loss as a python float.

        states:  (B, state_dim)       already on self.device
        actions: (B, chunk_size, 2)   already on self.device
        """
        
        loss = self.model.compute_loss(states, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _validate(self) -> float | None:
        """Mean loss over the validation loader, or None if there isn't one."""
        if self.val_loader is None:
            return None
        self.model.eval()
        total, count = 0.0, 0
        for states, actions in self.val_loader:
            states, actions = states.to(self.device), actions.to(self.device)
            total += self.model.compute_loss(states, actions).item() * states.shape[0]
            count += states.shape[0]
        return total / count

    def save_checkpoint(self, name: str, extra: dict | None = None) -> Path:
        """Save model weights, config and any extras (e.g. normalisation stats)."""
        path = self.run_dir / f"{name}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": asdict(self.cfg),
                "step": self.step,
                **self.ckpt_extra,
                **(extra or {}),
            },
            path,
        )
        return path

    def fit(self) -> None:
        for epoch in range(1, self.cfg.num_epochs + 1):
            self.model.train()
            epoch_loss, batches = 0.0, 0
            for states, actions in self.train_loader:
                states, actions = states.to(self.device), actions.to(self.device)
                loss = self._train_step(states, actions)
                self.step += 1
                epoch_loss += loss
                batches += 1
                if self.step % self.cfg.log_interval == 0:
                    print(f"step {self.step} | loss {loss:.4f}")

            row = {"epoch": epoch, "step": self.step, "train_loss": epoch_loss / batches}
            val_loss = self._validate()
            if val_loss is not None:
                row["val_loss"] = val_loss
            self.history.append(row)
            print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

        self.save_checkpoint("final")


def main() -> None:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    dataset = PushTChunkDataset(str(cfg.zarr_path), chunk_size=cfg.chunk_size)
    train_loader = build_dataloader(dataset, cfg.batch_size, shuffle=True, drop_last=True)

    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    model = MSEPolicy(state_dim, cfg.chunk_size * action_dim, cfg.hidden_dims)

    # Stored as tensors so the checkpoint loads without pickling numpy arrays.
    normalizer_stats = {
        "normalizer": {
            "state_mean": torch.from_numpy(dataset.mean_states),
            "state_std": torch.from_numpy(dataset.std_states),
            "action_mean": torch.from_numpy(dataset.mean_actions),
            "action_std": torch.from_numpy(dataset.std_actions),
        }
    }

    trainer = Trainer(cfg, model, train_loader, ckpt_extra=normalizer_stats)
    trainer.fit()


if __name__ == "__main__":
    main()
