
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class TrainConfig:
    # --- Data ---
    zarr_path: Path = Path(
        "/home/parikshat/MyWorkings/learning/imitation_learning/pushT/pusht/pusht_cchi_v7_replay.zarr"
    )
    # Number of future actions predicted per state (action chunk length).
    chunk_size: int = 8
    # Fraction of episodes held out for validation (split by episode, not by sample).
    val_fraction: float = 0.1

    # --- Model ---
    policy_type: Literal["mse", "flow"] = "mse"
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # Euler integration steps at sampling time (only used by the flow policy).
    flow_num_steps: int = 10

    # --- Optimization ---
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.0
    num_epochs: int = 400

    # --- Reproducibility / hardware ---
    seed: int = 42
    # "auto" picks cuda if available, otherwise cpu.
    device: Literal["cpu", "cuda"] = "cuda"

    # --- Logging / evaluation ---
    # Log the training loss every this many optimizer steps.
    log_interval: int = 100
    # Run a rollout evaluation every this many epochs.
    eval_interval: int = 10
    num_eval_episodes: int = 20
    # Results go to <output_dir>/<run_name>/.
    run_name: str = "MSEPolicy"
    output_dir: Path = Path("runs")
