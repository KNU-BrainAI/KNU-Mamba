import abc
import logging
import os
import pickle
import tempfile
from collections import deque
from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import wandb
from MambaVLA.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler

log = logging.getLogger(__name__)




from MambaVLA.utils.scaler import ActionScaler, MinMaxScaler, Scaler



class MambaVLA(nn.Module):
    def __init__(
        self,
        model: Any,
        obs_encoders: Any,
        language_encoders: Any,
        optimizer: Any,
        lr_scheduler: Any,
        action_dim: int,
        perception_seq_len: int,
        action_seq_len: int,
        cam_names: list[str],
        use_lr_scheduler: bool = True,
        consider_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
        sampling_steps: int = 50,
    ):
        super().__init__()

        self.device = device
        self.working_dir = os.getcwd()
        self.scaler = None

        # Initialize model and encoders
        self.img_encoder = obs_encoders.to(device)
        self.language_encoder = language_encoders.to(device)
        self.model = model.to(device)
        self.state_emb = nn.Linear(state_dim, latent_dim)

        self.cam_names = cam_names

        # for inference
        self.rollout_step_counter = 0
        self.action_seq_len = action_seq_len
        self.perception_seq_len = perception_seq_len

        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

        self.action_dim = action_dim

        self.consider_robot_states = consider_robot_states
        self.if_film_condition = if_film_condition


        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler

        self.use_lr_scheduler = use_lr_scheduler

        self.sampling_steps = sampling_steps

    def set_scaler(self, scaler):
        self.scaler = scaler

    def _input_embeddings(self, obs_dict):

        if "lang" in obs_dict:
            obs_dict["lang_emb"] = self.language_encoder(obs_dict["lang"]).float()

        lang_embed = obs_dict["lang_emb"]

        if self.cam_names is not None:

            B, T, C, H, W = obs_dict[f"{self.cam_names[0]}_image"].shape

            for camera in self.cam_names:
                obs_dict[f"{camera}_image"] = obs_dict[f"{camera}_image"].view(B * T, C, H, W)

            if self.if_film_condition:
                obs_embed = self.img_encoder(obs_dict, lang_embed)
            else:
                obs_embed = self.img_encoder(obs_dict)
        else:
            raise NotImplementedError("Either use point clouds or images as input.")

        if self.consider_robot_states and "robot_states" in obs_dict.keys():
            robot_states = obs_dict["robot_states"]
            robot_states = self.state_emb(robot_states)

            obs_embed = torch.cat([obs_embed, robot_states], dim=1)

        return obs_embed, lang_embed

    def reset(self):
        self.rollout_step_counter = 0
        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

    @torch.no_grad()
    def predict(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.obs_seq:
            for key in obs_dict.keys():
                self.obs_seq[key] = deque(maxlen=self.perception_seq_len)

        for key in obs_dict.keys():
            self.obs_seq[key].append(obs_dict[key])

            if key == "lang":
                continue
            obs_dict[key] = torch.concat(list(self.obs_seq[key]), dim=1)

            if obs_dict[key].shape[1] < self.perception_seq_len:
                pad = einops.repeat(
                    obs_dict[key][:, 0],
                    "b ... -> b t ...",
                    t=self.perception_seq_len - obs_dict[key].shape[1],
                )
                obs_dict[key] = torch.cat([pad, obs_dict[key]], dim=1)
                
        if self.rollout_step_counter == 0:
            self.eval()

            pred_action_seq = self(obs_dict)[:, :self.action_seq_len]
            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.action_seq_len:
            self.rollout_step_counter = 0

        return current_action

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        path = os.path.join(
            weights_path,
            "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth",
        )
        self.load_state_dict(torch.load(path, weights_only=True))
        log.info("Loaded pre-trained model")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        os.makedirs(store_path, exist_ok=True)
        path = os.path.join(
            store_path, "model_state_dict.pth" if sv_name is None else f"{sv_name}.pth"
        )

        state_dict = self.state_dict()

        fd, tmp_path = tempfile.mkstemp(dir=store_path, suffix=".tmp")
        os.close(fd)
        try:
            self._write_state_dict(state_dict, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        log.info(f"Model saved to: {store_path}")

    def _write_state_dict(self, state_dict: dict[str, torch.Tensor], path: str) -> None:
        try:
            torch.save(state_dict, path)
        except RuntimeError as err:
            log.warning(
                "torch.save failed with error '%s'. Retrying with legacy serialization.",
                err,
            )
            torch.save(state_dict, path, _use_new_zipfile_serialization=False)

    def store_model_scaler(self, store_path: str, sv_name=None) -> None:
        save_path = os.path.join(
            store_path, "model_scaler.pkl" if sv_name is None else sv_name
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.scaler, f)
        log.info(f"Model scaler saved to: {save_path}")

    def load_model_scaler(self, weights_path: str, sv_name=None) -> None:
        if sv_name is None:
            sv_name = "model_scaler.pkl"

        with open(os.path.join(weights_path, sv_name), "rb") as f:
            self.scaler = pickle.load(f)
        log.info("Loaded model scaler")

    def get_params(self):

        total_params = sum(p.numel() for p in self.parameters())

        wandb.log({"model parameters": total_params})

        log.info("The model has a total amount of {} parameters".format(total_params))

    @property
    def get_model_state_dict(self) -> dict:
        return self.state_dict()

    @property
    def get_scaler(self) -> Scaler:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return self.scaler

    @property
    def get_model_state(self) -> tuple[dict, Scaler]:
        if self.scaler is None:
            raise AttributeError("Scaler has not been set. Use set_scaler() first.")
        return (self.state_dict(), self.get_scaler)

    def recover_model_state(self, model_state, scaler):
        self.load_state_dict(model_state)
        self.set_scaler(scaler)

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        optim_groups = [
            {"params": self.model.model.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ]

        optim_groups.extend([
            {"params": self.img_encoder.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
        ])


        optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer_config.learning_rate,
                                      betas=self.optimizer_config.betas)

        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            scheduler = TriStageLRScheduler(optimizer, self.lr_scheduler)

            return optimizer, scheduler

        else:
            return optimizer

    def forward(self, obs_dict, actions=None):

        # with torch.no_grad():
        obs_embed, lang_embed = self._input_embeddings(obs_dict)

        if self.training and actions is not None:
            loss, _ = self.model(actions, obs_embed, lang_embed)

            return loss

        noise_actions = torch.randn((len(obs_embed), self.action_seq_len, self.action_dim), device=self.device)
        pred_act_seq = self.model.generate_actions(noise_actions, obs_embed, lang_embed, sample_steps=self.sampling_steps)

        return pred_act_seq
