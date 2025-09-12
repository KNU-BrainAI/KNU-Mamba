import torch
import torch.nn as nn
from typing import Any


class ActionFLowMatching(nn.Module):
    def __init__(
            self,
            backbones: Any,
            ln=False,
            device: str = 'cpu',
    ):
        super(ActionFLowMatching, self).__init__()

        self.model = backbones.to(device)
        self.ln = ln

    def forward(self, actions, state, lang_embed):
        batch_size = actions.size(0)
        if self.ln:
            noise_t = torch.randn((batch_size,)).to(actions.device)
            time_steps = torch.sigmoid(noise_t)
        else:
            time_steps = torch.rand((batch_size,)).to(actions.device)
        time_expanded = time_steps.view([batch_size, *([1] * len(actions.shape[1:]))])
        noise_actions = torch.randn_like(actions)
        interpolated_actions = (1 - time_expanded) * actions + time_expanded * noise_actions

        # for fitting the architecture inputs
        # velocity_pred = self.model(interpolated_actions, time_steps, cond)
        velocity_pred = self.model(states=state, actions=interpolated_actions, lang_cond=lang_embed, sigma=time_steps)

        batchwise_mse = ((noise_actions - actions - velocity_pred) ** 2).mean(dim=list(range(1, len(actions.shape))))
        time_loss_list = batchwise_mse.detach().cpu().reshape(-1).tolist()
        time_loss_pairs = [(time_val, loss_val) for time_val, loss_val in zip(time_steps, time_loss_list)]
        return batchwise_mse.mean(), time_loss_pairs

    @torch.no_grad()
    def generate_actions(self, noise_actions, state, lang_embed, null_cond=None, sample_steps=50, cfg=2.0):
        batch_size = noise_actions.size(0)
        step_size = 1.0 / sample_steps
        step_size = torch.tensor([step_size] * batch_size).to(noise_actions.device).view([batch_size, *([1] * len(noise_actions.shape[1:]))])
        samples = [noise_actions]
        for i in range(sample_steps, 0, -1):
            time_step = i / sample_steps
            time_step = torch.tensor([time_step] * batch_size).to(noise_actions.device)
            velocity_pred = self.model(states=state, actions=noise_actions, lang_cond=lang_embed, sigma=time_step)
            noise_actions = noise_actions - step_size * velocity_pred
            samples.append(noise_actions)
        return samples[-1]