# Imitation Learning on Push-T

Behavior cloning with action chunking on the Push-T task. A policy sees the
state `[agent x, agent y, block x, block y, block angle]` and predicts the next
8 target positions for the agent (cursor). Two policies are trained on the same
human demonstrations and compared:

- **MSE policy**: an MLP regressing the action chunk directly.
- **Flow matching policy**: a generative model that learns a velocity field from
  noise to the action chunk, which can represent several valid ways of doing
  the task instead of averaging them.

## Layout

```
pushT/
  config.py      TrainConfig dataclass: every setting for a run
  data.py        PushTChunkDataset: sliding-window (state, action chunk) pairs + normalisation
  model.py       MSEPolicy, FlowMatchingPolicy (+ SinusoidalPosEmb, TimeEncoder, Film), build_policy
  trainer.py     Trainer: training loop, validation hook, checkpointing (python trainer.py --policy ...)
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

## The two policies

**MSE.** `state (5) -> MLP -> 16 numbers` reshaped to 8 actions of 2 values, trained
with mean squared error against the demonstrated chunk.

**Flow matching.** Training blends the true (normalised) action chunk `x` with
Gaussian noise `n`: `x_t = (1 - t) * n + t * x`, with `t ~ U(0, 1)` per sample. The
network predicts the velocity `x - n` from `(x_t, state, t)`. The time is turned into
a sinusoidal embedding and a small MLP, and together with the state it conditions every
hidden layer through FiLM (a learned scale and shift). To act, it starts from noise
and takes 10 Euler steps along the predicted velocity, from `t = 0` to `t = 1`.

The two models are sized to have about the same number of parameters, so the
comparison is not about model size:

| Policy | Hidden widths | Parameters |
|---|---|---|
| MSE | (256, 256, 256) | 137,232 |
| Flow | (160, 160, 160, 160) | 139,280 |

## Train

Settings are in `pushT/config.py` (`TrainConfig`): chunk size 8, batch size 128,
AdamW with lr 3e-4, 400 epochs. Run from `pushT/`:

```
cd pushT
python trainer.py --policy mse     # writes runs/MSEPolicy/final.pt
python trainer.py --policy flow    # writes runs/FlowPolicy/final.pt
```

`--run_name NAME` overrides the run folder name. Each checkpoint holds the weights,
the config, the policy type, the layer widths and the normalisation statistics, so
evaluation can rebuild the model on its own. Training stops with `FileExistsError` if
the run folder already exists, so choose another `--run_name` or delete the old run.
The default device is CUDA, with a fallback to CPU.

## Evaluate

```
cd pushT
python evaluation.py --run_name MSEPolicy
python evaluation.py --run_name FlowPolicy
```

Options: `--episodes` (default 100) and `--render_episodes` (default 5; the first
N episodes are shown in a window, the rest run headless; use 0 for none). The
render window opens at `SDL_VIDEO_WINDOW_POS` (default `200,150`).

It runs the episodes in `gym_pusht` (seeds 0 to 99, at most 300 steps each). The
policy predicts a chunk of 8 actions and all 8 are executed before it is queried
again. It reports two numbers:

- **Mean max reward**: the average, over episodes, of the best reward reached. The
  reward is `coverage / 0.95` (capped at 1), where coverage is the fraction of the
  target area covered by the T. This is the main metric.
- **Success rate**: the share of episodes that reach more than 95% coverage. This is
  much stricter, since an episode that ends at 92% coverage scores about 0.97 on
  reward but counts as a failure here.

## Results

400 epochs each, 100 episodes per evaluation:

| Policy | Mean max reward | Success rate |
|---|---|---|
| MSE | 0.634 ± 0.387 | 10% |
| Flow matching | 0.876 ± 0.021 | 22.2% ± 3.6% |

How to read the numbers:

- MSE is deterministic, so repeated evaluations give the same result. The ± is
  the spread across the 100 episodes.
- Flow matching draws new noise at every query, so each evaluation differs. The
  figures are the mean ± standard deviation over 5 evaluations of the same checkpoint.
  Across those 5, the success rate was 18 to 27%, and two separate single
  evaluations gave 11% and 23%, so the success rate is noisy at 100 episodes.
  The reward is much steadier (0.84 to 0.90).
- Each policy was trained once, so these results do not include variation from
  the training seed.
- Per-episode flow rewards: of 100 episodes, 69 reached a reward of 0.9 or more, and 32
  stopped just short of the 95% coverage needed to count as a success. The median
  episode scored 0.948.
- The MSE policy is spread out (median 0.85, minimum 0.0). Flow matching raises the
  mean reward by about 0.24, which fits the idea that MSE averages between valid
  ways of pushing the block.
