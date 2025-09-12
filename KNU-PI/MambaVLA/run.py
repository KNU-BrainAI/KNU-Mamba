"""
Main training script using dataclass configuration instead of Hydra.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
import wandb

import multiprocessing as mp

# Set multiprocessing start method to 'spawn' to avoid CUDA issues
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass


import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import create_config, create_libero_object_config, create_libero_spatial_config, create_libero_goal_config
from configs.factory import create_model, create_trainer, create_simulation

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def set_seed_everywhere(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(benchmark_type: str = "libero_object", checkpoint_path: str | None = None) -> None:
    """
    Main training function.
    
    Args:
        benchmark_type: The task suite to use ('libero_object', 'libero_spatial', 'libero_goal')
    """
    
    # Create configuration based on task suite
    if benchmark_type == "libero_object":
        cfg = create_libero_object_config()
    elif benchmark_type == "libero_spatial":
        cfg = create_libero_spatial_config()
    elif benchmark_type == "libero_goal":
        cfg = create_libero_goal_config()
    else:
        cfg = create_config()
    
    set_seed_everywhere(cfg.seed)
    
    # Initialize wandb logger
    wandb_config = {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "group": cfg.group,
        "seed": cfg.seed,
        "benchmark_type": cfg.dataset.benchmark_type,
        "demos_per_task": cfg.dataset.demos_per_task,
        "chunck_size": cfg.chunck_size,
        "perception_seq_len": cfg.perception_seq_len,
        "action_seq_len": cfg.action_seq_len,
        "train_batch_size": cfg.train_batch_size,
        "epoch": cfg.epoch,
        "device": cfg.device,
        "len_embd": cfg.len_embd,
        "latent_dim": cfg.latent_dim,
        "action_dim": cfg.action_dim,
        "state_dim": cfg.state_dim,
    }
    
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        config=wandb_config
    )
    
    # Create output directory structure: runs/<benchmark_type>/<YYYYMMDD>/<HHMMSS>/
    import datetime
    now = datetime.datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    time_dir = now.strftime("%H-%M-%S")
    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    run_output_dir = os.path.join(output_root, cfg.dataset.benchmark_type, date_dir, time_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    # Create model and set its working_dir to the run-specific directory
    model = create_model(cfg)
    model.working_dir = run_output_dir
    
    # Create trainer and set its working_dir as well
    trainer = create_trainer(cfg)
    trainer.working_dir = run_output_dir
    
    # Get model parameters for logging
    model.get_params()

    # If a checkpoint is provided, load it and skip training
    if checkpoint_path is not None:
        # Set scaler from trainer to ensure inference works
        model.set_scaler(trainer.scaler)

        # Resolve checkpoint: file path or directory
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(state_dict)
            log.info(f"Loaded checkpoint from file: {checkpoint_path}")
        elif os.path.isdir(checkpoint_path):
            candidates = [
                os.path.join(checkpoint_path, "final_model.pth"),
                os.path.join(checkpoint_path, "model_state_dict.pth"),
            ]
            loaded = False
            for cand in candidates:
                if os.path.isfile(cand):
                    state_dict = torch.load(cand, weights_only=True)
                    model.load_state_dict(state_dict)
                    log.info(f"Loaded checkpoint from directory: {cand}")
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path} (looked for final_model.pth, model_state_dict.pth)")
    else:
        # Train the model if no checkpoint provided
        trainer.main(model)
    
    # Create simulation environment
    env_sim = create_simulation(cfg)
    env_sim.get_task_embs(trainer.trainset.tasks)
    
    # Test the model
    env_sim.test_model(model, cfg.model_cfg, epoch=cfg.epoch)
    
    log.info("Training done")
    log.info("state_dict saved in {}".format(model.working_dir))
    wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MambaVLA model")
    parser.add_argument(
        "--benchmark_type", 
        type=str, 
        default="libero_object",
        choices=["libero_object", "libero_spatial", "libero_goal"],
        help="Task suite to use for training"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint (.pth file or directory). If provided, skips training and evaluates with this checkpoint."
    )
    
    args = parser.parse_args()
    main(args.benchmark_type, args.checkpoint_path)
