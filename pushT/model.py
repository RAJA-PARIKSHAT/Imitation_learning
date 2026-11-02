import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 


class SinusoidalPosEmb(nn.Module):
    """Embed a scalar time t into a vector of sines and cosines at different frequencies.

    t: (B,) in [0, 1]  ->  (B, dim)

    t is multiplied by t_scale first. With t in [0, 1] and the slowest frequencies
    near 1/10000, the unscaled low-frequency dimensions would be almost constant.
    """

    def __init__(self, dim: int, t_scale: float = 1000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "dim must be even (half sines, half cosines)"
        self.dim = dim
        self.t_scale = t_scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        # frequencies fall geometrically from 1 down to about 1/10000
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        angles = (t.float() * self.t_scale)[:, None] * freqs[None, :]   # (B, half)
        return torch.cat([angles.sin(), angles.cos()], dim=-1)          # (B, dim)


class TimeEncoder(nn.Module):

    def __init__(self, in_dimension, emb_dimension, t_scale):
        super().__init__()

        self.sinusoidal_embedding = SinusoidalPosEmb(in_dimension, t_scale= t_scale)
        self.linear = nn.Linear(in_dimension, emb_dimension)


    def forward(self, t):

        sinusoidal_projection = self.sinusoidal_embedding(t)
        return F.relu(self.linear(sinusoidal_projection))
    

class Film(nn.Module):

    def __init__(self, cond_dim, in_channels):
        super().__init__()

        self.film_gen = nn.Linear(cond_dim, 2*in_channels)
        # start as the identity: gamma = 0 -> factor (1 + gamma) = 1, beta = 0
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x, cond):

        gamma, beta = self.film_gen(cond).chunk(2, dim = -1)
        return (1 + gamma) * x + beta
    

class FlowMatchingPolicy(nn.Module):

    def __init__(self, action_dim, 
                 chunk_size, 
                 state_dim, 
                 time_emb_dim,
                 hidden_states):
        super().__init__()

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_states = hidden_states

        # time embedding modules 

        self.time_embedder = TimeEncoder(in_dimension= 128, emb_dimension= 32, t_scale= 1000)

        # state conditioning projector 
        self.state_embedder = nn.Linear(state_dim, 16)

        in_concat = 16 + action_dim*chunk_size

        # ModuleList (not plain lists) so the layers are registered: they get trained,
        # moved by .to(device) and saved in the state_dict
        self.linear_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        self.linear_layers.append(nn.Linear(in_features= in_concat, out_features= self.hidden_states[0]))
        self.film_layers.append(Film(cond_dim= 32, in_channels= self.hidden_states[0]))
        
        for in_hidden, out_hidden in zip(self.hidden_states[:-1], self.hidden_states[1:]):
            self.linear_layers.append(nn.Linear(in_features= in_hidden + 16, out_features= out_hidden))
            self.film_layers.append(Film(cond_dim= 32, in_channels= out_hidden))

        self.out_layer = nn.Linear(self.hidden_states[-1], action_dim*chunk_size)

    def forward(self, x, state, t):

        state_cond = self.state_embedder(state)
        t_cond = self.time_embedder(t)

        for i in range(len(self.linear_layers)):
            x = torch.cat([x, state_cond], dim = -1)
            x = self.linear_layers[i](x)
            x = self.film_layers[i](x, t_cond)
            x = F.relu(x)

        return self.out_layer(x)

    def compute_loss(self, state, actions):

        # (B, chunk_size, action_dim) -> (B, chunk_size * action_dim)
        x = actions.reshape(actions.shape[0], -1)
        device = x.device

        x_0 = torch.randn_like(x)
        t = torch.rand((x.shape[0], ), device = device)    # (B,): conditioning input
        t_blend = t[:, None]                               # (B, 1): broadcasts over the chunk

        x_t = (1 - t_blend)*x_0 + t_blend*x
        gt_vel = x - x_0
        pred_vel = self.forward(x_t, state, t)

        return F.mse_loss(gt_vel, pred_vel)

    
    @torch.no_grad()
    def sample_actions(self, state, num_steps: int = 10):
        
        
        single = state.dim() < 2
        if single:
            state = state[None]

        x_t = torch.randn((state.shape[0], self.action_dim * self.chunk_size), device = state.device)
        dt = 1.0 / num_steps
        timesteps = torch.linspace(0, 1.0 - dt, steps= num_steps, device= state.device)

        for i in range(num_steps):

            t__curr = torch.full((state.shape[0],), timesteps[i], device= state.device).float()

            pred_velocity = self.forward(x_t, state, t__curr)

            x_t = x_t + pred_velocity*dt

        actions = x_t.reshape(state.shape[0], self.chunk_size, self.action_dim)

        return actions[0] if single else actions
    
class MSEPolicy(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_states):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_states = hidden_states
        layers = []

        layers.extend([nn.Linear(in_features= self.input_dim, out_features= self.hidden_states[0]), nn.ReLU()])

        for in_hidden, out_hidden in zip(self.hidden_states[:-1], self.hidden_states[1:]):
            layers.extend([nn.Linear(in_features= in_hidden, out_features= out_hidden), nn.ReLU()])

        layers.append(nn.Linear(self.hidden_states[-1], self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)
    
    def compute_loss(self, states, actions):
        predicted = self.forward(states)

        return F.mse_loss(actions.reshape(actions.shape[0], -1), predicted)

    @torch.no_grad()
    def sample_actions(self, state, num_steps: int = 10):
        # num_steps is unused: MSE predicts in one pass. It only keeps the same
        # interface as the flow policy.
        single = state.dim() < 2
        if single:
            state = state[None]

        actions = self.forward(state).reshape(state.shape[0], -1, 2)

        return actions[0] if single else actions


def build_policy(policy_type, state_dim, action_dim, chunk_size, hidden_dims, time_emb_dim=32):
    """The one place that knows how to construct each policy (training and evaluation use it)."""
    if policy_type == "mse":
        return MSEPolicy(state_dim, chunk_size * action_dim, hidden_dims)
    if policy_type == "flow":
        return FlowMatchingPolicy(action_dim, chunk_size, state_dim, time_emb_dim, hidden_dims)
    raise ValueError(f"Unknown policy type: {policy_type}")



