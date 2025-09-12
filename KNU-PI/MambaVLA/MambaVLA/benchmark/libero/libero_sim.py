import logging
import os
import cv2
import random
import numpy as np
import torch
import wandb
import hydra
import multiprocessing as mp
# from .base_sim import BaseSim
# from libero.libero.envs import *
from tqdm import tqdm
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from colorama import init, Fore, Style, Back

log = logging.getLogger(__name__)

# Initialize colorama for cross-platform colored output
init(autoreset=True)


def print_colored_success_array(success_tensor):
    """Print success array with color coding: green for success (1), red for failure (0)"""
    success_np = success_tensor.detach().cpu().numpy()
    print(f"\n{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}║                    SUCCESS MATRIX                           ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    for i in range(success_np.shape[0]):
        row_str = f"{Fore.YELLOW}Task {i:2d}:{Style.RESET_ALL} "
        for j in range(success_np.shape[1]):
            if success_np[i, j] == 1:
                row_str += f"{Fore.GREEN}{Back.GREEN}{Style.BRIGHT} ✓ {Style.RESET_ALL} "
            else:
                row_str += f"{Fore.RED}{Back.RED}{Style.BRIGHT} ✗ {Style.RESET_ALL} "
        print(row_str)
    print()


def print_progress_header(total_tasks, total_episodes, use_multiprocessing, render_enabled):
    """Print a clear header showing the evaluation setup"""
    print(f"\n{Fore.MAGENTA}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}║                    EVALUATION SETUP                          ║{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Task Suite:{Style.RESET_ALL} {Fore.YELLOW}{total_tasks} tasks{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Episodes per Task:{Style.RESET_ALL} {Fore.YELLOW}{total_episodes}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Total Evaluations:{Style.RESET_ALL} {Fore.YELLOW}{total_tasks * total_episodes}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Multiprocessing:{Style.RESET_ALL} {Fore.GREEN if use_multiprocessing else Fore.RED}{'Enabled' if use_multiprocessing else 'Disabled'}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}• Real-time Rendering:{Style.RESET_ALL} {Fore.GREEN if render_enabled else Fore.RED}{'Enabled' if render_enabled else 'Disabled'}{Style.RESET_ALL}")
    print()


def print_evaluation_summary(success_rate, average_success, num_tasks):
    """Print a clear summary of evaluation results"""
    print(f"\n{Fore.GREEN}{Style.BRIGHT}╔══════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}║                    EVALUATION RESULTS                        ║{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{Style.BRIGHT}╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}Overall Average Success Rate:{Style.RESET_ALL} {Fore.YELLOW}{average_success:.3f}{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}Per-Task Success Rates:{Style.RESET_ALL}")
    
    for num in range(num_tasks):
        success_val = success_rate[num].item()
        color = Fore.GREEN if success_val >= 0.8 else Fore.YELLOW if success_val >= 0.5 else Fore.RED
        print(f"  {Fore.CYAN}Task {num:2d}:{Style.RESET_ALL} {color}{success_val:.3f}{Style.RESET_ALL}")
    print()


def log_episode_progress(completed_success, completed_lengths, average_success, average_episode_length, current_count, total_runs, current_task=None, current_episode=None, task_name=None):
    """Log episode progress with inline updates"""
    # Calculate percentage
    progress_pct = (current_count / total_runs) * 100
    
    # Create progress bar
    bar_length = 30
    filled_length = int(bar_length * current_count // total_runs)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Build the progress line
    progress_line = f"{Fore.CYAN}Progress:{Style.RESET_ALL} [{bar}] {current_count:3d}/{total_runs} ({progress_pct:5.1f}%)"
    
    # Add task/episode info if available
    if current_task is not None and current_episode is not None:
        task_info = f" | {Fore.MAGENTA}Task:{Style.RESET_ALL} {current_task:2d} {Fore.MAGENTA}Episode:{Style.RESET_ALL} {current_episode:2d}"
    else:
        task_info = ""
    
    # Add task name if available
    if task_name is not None:
        task_name_info = f" | {Fore.BLUE}Task Name:{Style.RESET_ALL} {Fore.WHITE}{task_name}{Style.RESET_ALL}"
    else:
        task_name_info = ""
    
    # Add success metrics
    success_line = f" | {Fore.GREEN}Success:{Style.RESET_ALL} {average_success:.3f} | {Fore.YELLOW}Length:{Style.RESET_ALL} {average_episode_length:.1f}"
    
    # Print progress line (without carriage return for now)
    print(f"{progress_line}{task_info}{task_name_info}{success_line}")
    
    # If this is the last episode, add a separator
    if current_count == total_runs:
        print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")





def safe_display_image(img, window_name, render_enabled):
    """Safely display image with error handling for headless environments"""
    if not render_enabled:
        return
        
    try:
        # Check if we have a display available
        if 'DISPLAY' not in os.environ or os.environ.get('DISPLAY') == '':
            return
            
        cv2.imshow(window_name, img)
        cv2.waitKey(1)  # refresh window
    except Exception as e:
        # Silently fail if display is not available
        pass


def safe_destroy_window(window_name, render_enabled):
    """Safely destroy display window with error handling"""
    if not render_enabled:
        return
        
    try:
        if 'DISPLAY' in os.environ and os.environ.get('DISPLAY') != '':
            cv2.destroyWindow(window_name)
    except Exception as e:
        # Silently fail if display is not available
        pass


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def process_image_input(img_tensor):
    # return (img_tensor / 255. - 0.5) * 2.
    return img_tensor / 255.


class MultiTaskSim():
    def __init__(self,
                 rollouts,
                 max_step_per_episode,
                 benchmark_type: str,
                 use_eye_in_hand: bool,
                 seed,
                 device,
                 render_image,
                 n_cores,
                 use_multiprocessing=True,
                 save_video=False,
                 save_video_dir=None):
        # super().__init__(seed, device, render, n_cores)

        self.seed = seed
        self.device = device
        self.render_image = render_image
        self.n_cores = n_cores

        # according to the task_id, load the corresponding bddl file
        self.benchmark_type = benchmark_type

        self.use_eye_in_hand = use_eye_in_hand
        self.render_image = render_image
        self.save_video = save_video
        self.save_video_dir = save_video_dir
        self.rollouts = rollouts
        self.max_step_per_episode = max_step_per_episode

        self.success_rate = 0
        self.use_multiprocessing = use_multiprocessing

    def reverse_rgb_channels(self, test_img):

        test_img = test_img[::-1, ::-1, :]
        # cv2.imshow("test_img", test_img)
        # cv2.waitKey(0)

        return np.ascontiguousarray(test_img)

    def eval_model(self,
                   contexts,
                   context_ind,
                   success,
                   episode_lengths,
                   pid,
                   cpu_set,
                   counter,
                   all_runs,
                   model=None,
                   model_config=None,
                   model_states=None):
        # Only set CPU affinity if using multiprocessing
        # if self.use_multiprocessing:
        #     print(os.getpid(), cpu_set)
        #     assign_process_to_cpu(os.getpid(), cpu_set)

        # Handle model initialization based on input type
        if model_config is not None:
            # Case 1: Initialize model from config and states
            assert model_states is not None, "model_states must be provided when using model_config"
            model = hydra.utils.instantiate(model_config)
            model.recover_model_state(
                model_states['model'],
                model_states['scaler']
            )
            # Ensure the freshly instantiated model is on the desired device
            model = model.to(self.device)
        else:
            # Case 2: Use provided model directly
            assert model is not None, "Either model or (model_config + states) must be provided"
            # Move the provided model to the correct device (CPU / CUDA)
            model = model.to(self.device)

        # print(contexts)

        for i, context in enumerate(contexts):

            benchmark_type = benchmark.get_benchmark_dict()[self.benchmark_type]()

            task_bddl_file = benchmark_type.get_task_bddl_file_path(context)

            file_name = os.path.basename(task_bddl_file).split('.')[0]

            task_emb = self.task_embs[file_name].to(self.device).unsqueeze(0)

            # goal_images = self.goal_dicts[file_name]
            # goal_image = random.choice(goal_images)

            init_states = benchmark_type.get_task_init_states(context)

            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128
            }

            env = OffScreenRenderEnv(**env_args)

            model.reset()
            env.seed(self.seed)
            env.reset()
            obs = env.set_init_state(init_state=init_states[context_ind[i]])

            # dummy actions all zeros for initial physics simulation
            dummy = np.zeros(7)
            dummy[-1] = -1.0  # set the last action to -1 to open the gripper
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            # Setup video recording with task-specific folders
            task_video_dir = None
            if self.save_video and self.save_video_dir is not None:
                os.makedirs(self.save_video_dir, exist_ok=True)
                task_video_dir = os.path.join(self.save_video_dir, f"{self.benchmark_type}", "videos", f"{file_name}")
    
            # Print task name in a box format
            task_name_length = len(file_name)
            box_width = max(50, task_name_length + 10)
            print(f"\n{Fore.CYAN}{Style.BRIGHT}╔{'═' * box_width}╗{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}║{' ' * ((box_width - task_name_length) // 2)}{Fore.WHITE}{Style.BRIGHT}{file_name}{Style.RESET_ALL}{' ' * ((box_width - task_name_length + 1) // 2)}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{Style.BRIGHT}╚{'═' * box_width}╝{Style.RESET_ALL}\n")
            
            video_writer = None
            if task_video_dir is not None:
                os.makedirs(task_video_dir, exist_ok=True)
                save_path = os.path.join(task_video_dir, f"episode_{context_ind[i]}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') #type: ignore
                video_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (1280, 800))

            # multiprocessing simulation
            for j in range(self.max_step_per_episode):
                agentview_rgb = torch.from_numpy(obs["agentview_image"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.
                eye_in_hand_rgb = torch.from_numpy(obs["robot0_eye_in_hand_image"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.

                joint_state = obs["robot0_joint_pos"]
                gripper_state = obs["robot0_gripper_qpos"]

                robot_states = torch.from_numpy(np.concatenate([joint_state, gripper_state], axis=-1)).to(self.device).float().unsqueeze(0).unsqueeze(0)

                # Record frame for video and display
                img = None
                if video_writer is not None or self.render_image:
                    img = env.sim.render(camera_name="frontview", width=1280, height=800)[..., ::-1]
                    img = np.flip(img, axis=0)
                    if video_writer is not None:
                        video_writer.write(img)
                
                # Real-time display if requested (with safe error handling)
                if self.render_image and img is not None:
                    window_name = f"frontview_{pid}"
                    safe_display_image(img, window_name, self.render_image)

                # img = env.sim.render(camera_name="frontview", width=1280, height=800)[..., ::-1]
                # img = np.flip(img, axis=0)
                # cv2.imwrite(os.path.join(save_path, f"agentview_{context}_{context_ind[i]}_{j}.png"), img)

                # agentview_rgb = self.reverse_rgb_channels(agentview_rgb)
                # eye_in_hand_rgb = self.reverse_rgb_channels(eye_in_hand_rgb)

                obs_dict = {"agentview_image": agentview_rgb,
                            "eye_in_hand_image": eye_in_hand_rgb,
                            "lang_emb": task_emb,
                            "robot_states": robot_states}

                action = model.predict(obs_dict).cpu().numpy()
                obs, r, done, _ = env.step(action)

                # if self.render_image:
                # env.render()

                if r == 1:
                    success[context, context_ind[i]] = r
                    episode_lengths[context, context_ind[i]] = j + 1
                    print(f"{Fore.GREEN}Task {context}, Episode {context_ind[i]}: SUCCESS at step {j+1}{Style.RESET_ALL}")
                    break
                    
            if success[context, context_ind[i]] == 0:
                episode_lengths[context, context_ind[i]] = self.max_step_per_episode
                print(f"{Fore.RED}Task {context}, Episode {context_ind[i]}: FAILED after {self.max_step_per_episode} steps{Style.RESET_ALL}")

            # Release video writer
            if video_writer is not None:
                video_writer.release()
            # Close the display window if open (with safe error handling)
            window_name = f"frontview_{pid}"
            safe_destroy_window(window_name, self.render_image)

            if hasattr(counter, 'get_lock'):  # If it's a multiprocessing Value
                with counter.get_lock():
                    counter.value += 1
                    current_count = counter.value
            else:  # If it's a simple object with value attribute (single process)
                counter.value += 1
                current_count = counter.value

            mask = episode_lengths.flatten() != 0
            completed_success = success.flatten()[mask]
            completed_lengths = episode_lengths.flatten()[mask]
            average_success = torch.mean(completed_success).item()
            average_episode_length = torch.mean(completed_lengths).item()
            
            # Use the new structured logging function with inline updates
            if hasattr(counter, 'get_lock'):  # If it's a multiprocessing Value
                current_count = counter.value
            else:  # If it's a simple object with value attribute (single process)
                current_count = counter.value
                
            log_episode_progress(completed_success, completed_lengths, average_success, average_episode_length, current_count, all_runs, context, context_ind[i], file_name)

            env.close()

    def get_task_embs(self, task_embs):
        self.task_embs = task_embs

    def test_model(self, model, model_config, cpu_set=None, epoch=None):
        logging.info("Start testing model on {} tasks".format(self.benchmark_type))

        # Check if we're in a headless environment and adjust render setting
        if 'DISPLAY' not in os.environ or os.environ.get('DISPLAY') == '':
            if self.render_image:
                log.warning(f"{Fore.YELLOW}No display detected - disabling real-time rendering for headless execution{Style.RESET_ALL}")
                self.render_image = False
            log.info(f"{Fore.CYAN}Note: Video recording will continue to work in headless mode{Style.RESET_ALL}")
        else:
            log.info(f"{Fore.GREEN}Display detected - real-time rendering enabled{Style.RESET_ALL}")

        # If evaluating on GPU, warn about multiprocessing but allow it if explicitly set
        if isinstance(self.device, str) and "cuda" in self.device and self.use_multiprocessing:
            log.warning(f"{Fore.YELLOW}CUDA device detected with multiprocessing enabled - this may cause GPU memory issues!{Style.RESET_ALL}")
            log.warning(f"{Fore.YELLOW}Consider using CPU evaluation for better multiprocessing performance.{Style.RESET_ALL}")

        if cpu_set is None:
            num_cpu = self.n_cores
            cpu_set = [i for i in range(num_cpu)]
        else:
            num_cpu = len(cpu_set)
        
        if self.benchmark_type == "libero_90":
            num_tasks = 50 # changed from 90 to 50
        else:
            num_tasks = 10
            
        # Print evaluation setup header (after num_tasks is defined)
        print_progress_header(num_tasks, self.rollouts, self.use_multiprocessing, self.render_image)
        
        if self.use_multiprocessing:
            log.info(f"{Fore.CYAN}Multiprocessing:{Style.RESET_ALL} {Fore.GREEN}Enabled with {num_cpu} CPUs{Style.RESET_ALL}")
        else:
            log.info(f"{Fore.CYAN}Multiprocessing:{Style.RESET_ALL} {Fore.RED}Disabled - running on 1 CPU{Style.RESET_ALL}")

        success = torch.zeros([num_tasks, self.rollouts]).share_memory_()
        episode_lengths = torch.zeros([num_tasks, self.rollouts]).share_memory_()
        all_runs = num_tasks * self.rollouts
        
        # # Debug: Print initial tensor state
        # print(f"{Fore.YELLOW}Debug: Created success tensor with shape {success.shape}{Style.RESET_ALL}")
        # print(f"{Fore.YELLOW}Debug: Initial success tensor: {success}{Style.RESET_ALL}")

        contexts = np.arange(num_tasks)
        contexts = np.repeat(contexts, self.rollouts)

        context_ind = np.arange(self.rollouts)
        context_ind = np.tile(context_ind, num_tasks)

        if not self.use_multiprocessing:
            # Single process execution
            counter = type('Counter', (), {'value': 0})()  # Simple counter object
            
            print(f"\n{Fore.CYAN}Starting evaluation with {all_runs} total episodes...{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")

            self.eval_model(
                contexts=contexts,
                context_ind=context_ind,
                success=success,
                episode_lengths=episode_lengths,
                pid=0,
                cpu_set=set(cpu_set),
                counter=counter,
                all_runs=all_runs,
                model=model
            )
        else:
            repeat_num = all_runs // num_cpu
            repeat_res = all_runs % num_cpu

            workload_array = np.ones([num_cpu], dtype=int)
            workload_array[:repeat_res] += repeat_num
            workload_array[repeat_res:] = repeat_num

            assert np.sum(workload_array) == all_runs

            ind_workload = np.cumsum(workload_array)
            ind_workload = np.concatenate([[0], ind_workload])
            ###################################################################
            ctx = mp.get_context('spawn')
            processes_list = []

            all_runs = num_tasks * self.rollouts
            counter = ctx.Value('i', 0) #create a shared counter for progress bar
            
            # Create shared memory state dictionaries for all models
            model_states = model.get_model_state
            shared_states = {
                'model': {},
                'scaler': model_states[1]  # Assuming scaler is the 4th element
            }
    
            # Share memory for each state dictionary
            for key, tensor in model_states[0].items():
                shared_states['model'][key] = tensor.share_memory_()

            print(f"\n{Fore.CYAN}Starting multiprocessing evaluation with {all_runs} total episodes across {self.n_cores} processes...{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'─' * 80}{Style.RESET_ALL}")
            
            for i in range(self.n_cores):
                p = ctx.Process(target=self.eval_model,
                                kwargs={  # Now passing single parameter
                                    "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                                    "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                                    "success": success,
                                    "episode_lengths": episode_lengths,
                                    "pid": i,
                                    "cpu_set": set(cpu_set[i:i + 1]),
                                    "counter": counter,
                                    "all_runs": all_runs,
                                    "model": None,
                                    "model_config": model_config,
                                    "model_states": shared_states,
                                },
                                )
                p.start()
                processes_list.append(p)
            
            # Wait for all processes to complete
            [p.join() for p in processes_list]

        success_rate = torch.mean(success, dim=-1)
        average_success = torch.mean(success_rate).item()

        # Ensure we have a clean line before showing results
        print()  # Add newline to separate from progress line
        
        # # Print comprehensive results with colors and structure
        # print(f"\n{Fore.CYAN}Debug: Success tensor shape: {success.shape}{Style.RESET_ALL}")
        # print(f"{Fore.CYAN}Debug: Success tensor contents: {success}{Style.RESET_ALL}")
        # print(f"{Fore.CYAN}Debug: Success rate shape: {success_rate.shape}{Style.RESET_ALL}")
        # print(f"{Fore.CYAN}Debug: Success rate contents: {success_rate}{Style.RESET_ALL}")
        
        print_colored_success_array(success)
        print_evaluation_summary(success_rate, average_success, num_tasks)

        # Log to wandb
        custom_step = f"{epoch}_custom_step"
        wandb.define_metric(custom_step)
        wandb.define_metric(f"{epoch}_tasks_success", step_metric=custom_step)

        for num in range(num_tasks):
            wandb.log({custom_step: num,
                       f"{epoch}_tasks_success": success_rate[num].item()
                       })

        wandb.log({f"epoch{epoch}_average_success": average_success})
        
        # Final summary log
        print()  # Add space after progress line
        log.info(f"{Fore.GREEN}{Style.BRIGHT}══════════════════════════════════════════════════════════════{Style.RESET_ALL}")
        log.info(f"{Fore.GREEN}{Style.BRIGHT}EVALUATION COMPLETE - Final Average Success Rate: {average_success:.3f}{Style.RESET_ALL}")
        log.info(f"{Fore.GREEN}{Style.BRIGHT}══════════════════════════════════════════════════════════════{Style.RESET_ALL}")