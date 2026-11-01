# Imitation Learning on Push-T

Behavior cloning with action chunking on the Push-T task. A policy sees the
state `[agent x, agent y, block x, block y, block angle]` and predicts the next
8 target positions for the agent (cursor). This repo currently has the MSE
policy; a flow matching policy is planned next, to compare against it on the
multimodal human demonstrations.

## Layout

```
pushT/
  config.py      TrainConfig dataclass: every setting for a run
  data.py        PushTChunkDataset: sliding-window (state, action chunk) pairs + normalisation
  model.py       MSEPolicy: MLP from state to a flattened action chunk
  trainer.py     Trainer: training loop, validation hook, checkpointing (run with python trainer.py)
  evaluation.py  Rollouts in gym_pusht and the evaluation metrics
```

## Setup

Python 3.10+ with:

```
pip install torch numpy zarr gymnasium gym-pusht "pymunk<7"
```

### Data

Download the Push-T demonstrations from the Diffusion Policy project:
https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip

Extract it so the Zarr store is at
`pushT/pusht/pusht_cchi_v7_replay.zarr`, or point `zarr_path` in
`pushT/config.py` to wherever it is. The dataset is 206 human episodes
(25,650 steps). It is git-ignored.

## Train the MSE policy

All settings are in `pushT/config.py` (`TrainConfig`): chunk size 8, hidden
layers (256, 256, 256), batch size 128, AdamW with lr 3e-4, 400 epochs.

```
cd pushT
python trainer.py
```

This writes `pushT/runs/MSEPolicy/final.pt`, containing the weights, the config
and the normalisation statistics. The run folder is named by `run_name` in the
config. Training stops with `FileExistsError` if that folder already exists, so
change `run_name` or delete the old run before training again. The default
device is CUDA, with a fallback to CPU.

## Evaluate

```
cd pushT
python evaluation.py
```

This loads `runs/MSEPolicy/final.pt` and runs 100 episodes in `gym_pusht`
(seeds 0 to 99, at most 300 steps each). The policy predicts a chunk of 8
actions and all 8 are executed before it is queried again. It reports:

- mean (and std) of the per-episode maximum reward, which measures how much of
  the target area the T covers
- success rate, from the environment's `is_success` flag

To watch episodes in a window, call `evaluate(..., render=True)`.

### Result (MSE baseline)

400 epochs, 100 episodes:

| Metric | Value |
|---|---|
| Mean max reward | 0.634 ± 0.387 |
| Success rate | 10% |

The per-episode scores are very spread out (median 0.85, minimum 0.0), which is
what an MSE policy trained on multimodal demonstrations tends to do.
