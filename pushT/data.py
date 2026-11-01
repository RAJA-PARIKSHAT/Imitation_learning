import numpy as np
import torch
import zarr
from torch.utils.data import Dataset



class PushTChunkDataset(Dataset):
    """(state, action_chunk) pairs from Push-T demos via a sliding window.

    Example item: state (5,), action_chunk (chunk_size, 2).
    """

    def __init__(self, zarr_path: str, chunk_size: int = 8) -> None:

        root = zarr.open(zarr_path, mode = "r")

        self.states  = root["data"]["state"][:]
        self.actions = root["data"]["action"][:]
        self.episode_ends = root["meta"]["episode_ends"][:]
        self.chunk_size = chunk_size

        last_episode = 0
        valid_indices = []
        for episode_end in self.episode_ends:
            valid_indices.append(np.arange(last_episode, episode_end - self.chunk_size + 1))
            last_episode = episode_end

        self.valid_indices = np.concatenate(valid_indices)
        self._compute_mean_and_std()

    def _compute_mean_and_std(self):

        self.mean_states = self.states.mean(axis = 0)
        self.std_states  = self.states.std(axis = 0)

        self.mean_actions = self.actions.mean(axis = 0)
        self.std_actions = self.actions.std(axis = 0) + 1e-8

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Raw state (..., 5) -> zero-mean, unit-std state."""
        return (state - self.mean_states) / self.std_states

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Raw actions (..., 2) -> zero-mean, unit-std actions."""
        return (actions - self.mean_actions) / self.std_actions

    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Inverse of normalize_actions: model output -> pixel coordinates for the env."""
        return actions * self.std_actions + self.mean_actions

    def __len__(self) -> int:
        return len(self.valid_indices)
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        index = self.valid_indices[idx]

        state = self.states[index]
        state = (state - self.mean_states) / self.std_states

        actions = self.actions[index:index+self.chunk_size]
        actions = (actions - self.mean_actions) / self.std_actions

        
        return torch.from_numpy(state).float(), torch.from_numpy(actions).float()