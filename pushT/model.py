import torch 
import torch.nn as nn
import torch.nn.functional as F

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
