import torch
import torch.nn as nn
import einops
import math
from typing import Optional, Any
from torch.nn import functional as F
import logging



logger = logging.getLogger(__name__)



class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = 1000 * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half) / half
        ).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    # @torch.compile()
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            dtype=next(self.parameters()).dtype
        )
        t_emb = self.mlp(t_freq)

        if len(t_emb.shape) == 2:
            t_emb = einops.rearrange(t_emb, 'b d -> b 1 d')

        return t_emb


class MambaVLAPolicy(nn.Module):
    def __init__(
            self,
            encoder: Any,
            latent_dim: int,
            action_dim: int,
            lang_emb_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            lang_tok_len: int,
            obs_tok_len: int,
            action_seq_len: int,
            linear_output: bool = False,
            use_ada_conditioning: bool = False,
            use_pos_emb: bool = True
    ):
        super().__init__()

        self.encoder = encoder

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            lang_tok_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = lang_tok_len + obs_tok_len + action_seq_len

        # linear embedding for the state
        self.tok_emb = nn.Linear(latent_dim, embed_dim)

        # linear embedding for the goal
        self.lang_emb = nn.Linear(lang_emb_dim, embed_dim)

        # linear embedding for the action
        self.action_emb = nn.Linear(action_dim, embed_dim)

        self.sigma_emb = TimeEmbedding(embed_dim)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            # position embedding
            self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))

        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        self.action_dim = action_dim
        self.obs_dim = latent_dim
        self.embed_dim = embed_dim

        self.lang_tok_len = lang_tok_len
        self.obs_tok_len = obs_tok_len
        self.action_seq_len = action_seq_len

        self.use_ada_conditioning = use_ada_conditioning

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
            self,
            states,
            actions,
            lang_cond,
            sigma
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        # t for the states does not mean the time, but the number of inputs tokens
        b, t, dim = states.size()
        _, t_a, _ = actions.size()

        if self.goal_conditioned:
            lang_embed = self.lang_emb(lang_cond)
            if self.use_pos_emb:
                lang_embed += self.pos_emb[:, :self.lang_tok_len, :]
            goal_x = self.drop(lang_embed)

        state_embed = self.tok_emb(states)
        if self.use_pos_emb:
            state_embed += self.pos_emb[:, self.lang_tok_len:(self.lang_tok_len + t), :]
        state_x = self.drop(state_embed)

        action_embed = self.action_emb(actions)
        if self.use_pos_emb:
            action_embed += self.pos_emb[:, (self.lang_tok_len + t):(self.lang_tok_len + t + t_a), :]
        action_x = self.drop(action_embed)

        emb_t = self.sigma_emb(sigma)

        if self.goal_conditioned:
            input_seq = torch.cat([emb_t, goal_x, state_x, action_x], dim=1)
        else:
            input_seq = torch.cat([emb_t, state_x, action_x], dim=1)

        if self.use_ada_conditioning:
            encoder_output = self.encoder(input_seq, emb_t)
        else:
            encoder_output = self.encoder(input_seq)

        pred_actions = self.action_pred(encoder_output[:, -self.action_seq_len:, :])

        return pred_actions
