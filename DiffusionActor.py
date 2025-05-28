# diffusion_actor.py
import torch
import torch.nn as nn
from DiffusionTransformer import DiffusionTransformer

class DiffusionActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, diffusion_steps=10):
        super().__init__()
        self.max_action = max_action
        self.diffusion_steps = diffusion_steps
        self.denoiser = DiffusionTransformer(state_dim, action_dim)

    def forward(self, state):
        batch_size = state.size(0)
        action = torch.randn(batch_size, self.denoiser.net[-1].out_features).to(state.device)

        for t in reversed(range(self.diffusion_steps)):
            t_norm = torch.full((batch_size, 1), t / self.diffusion_steps, device=state.device)
            noise_pred = self.denoiser(state, action, t_norm)
            action = action - noise_pred / self.diffusion_steps

        return action.clamp(-self.max_action, self.max_action)

    def generate_noisy_data(self, state, action):
        t_int = torch.randint(0, self.diffusion_steps, (state.size(0), 1), device=state.device)
        t = t_int / self.diffusion_steps
        noise = torch.randn_like(action)
        noisy_action = action + noise * t
        return noisy_action, noise, t
