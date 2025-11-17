"""MambaVLA package API."""

from .api.train_vla import train_vla
from .api.infer_vla import infer_vla

# Re-export core classes from submodules (compatible with Hydra config)
from .MambaVLA import MambaVLA  # type: ignore
from .MambaVLA.main import Trainer  # type: ignore
from .MambaVLA.policy.flowmatching import ActionFLowMatching  # type: ignore
from .MambaVLA.policy.policy import MambaVLAPolicy  # type: ignore
from .MambaVLA.backbones.clip.clip_img_global_encoder import CLIPImgEncoder  # type: ignore
from .MambaVLA.backbones.clip.clip_lang_encoder import LangClip  # type: ignore
from .MambaVLA.backbones.multi_img_obs_encoder import MultiImageObsEncoder  # type: ignore
from .MambaVLA.backbones.resnet.resnets import ResNetEncoder  # type: ignore
from .MambaVLA.mamba.mamba import MixerModel as MambaModel  # type: ignore

__all__ = [
    "train_vla",
    "infer_vla",
    "MambaVLA",
    "Trainer",
    "ActionFLowMatching",
    "MambaVLAPolicy",
    "CLIPImgEncoder",
    "LangClip",
    "MultiImageObsEncoder",
    "ResNetEncoder",
    "MambaModel",
]

