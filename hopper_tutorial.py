import multiprocessing
from collections import defaultdict 
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from rewardModel import *
from copy import deepcopy
import os
import csv
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name",
    type=str,
    default=None
)

parser.add_argument(
    "--mode",
    type=str,
    default = "default",
    choices = ["llm","baseline", "default"]
)

parser.add_argument(
    "--teacher_model_path",
    type=str,
    default=None
)

parser.add_argument(
    "--score_mean",
    type=float,
    default = 0.0
)

parser.add_argument(
    "--score_std",
    type=float,
    default=1.0
)

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
device=torch.device("cpu")

MODE= ""

num_cells = 64  # number of cells in each layer i.e. output dim.
lr = 4e-4
max_grad_norm = 1.0


frames_per_batch = 2048
# For a complete training, bring the number of frames up to 1M
total_frames = 200000



sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

configure = {
    "conv_filters": [],
    "conv_activation": True,
    "fc_layer_sizes": [[2,256],[256,256],[256,256],[256,1]],
    "clip_at_last": "",
    "clip_scale": 1,
    } 

teacher_model = RewardModel(config=configure)
teacher_model.eval()

def main(args, csv_file_path):
    mode = args.mode
    if mode == "llm" or mode == "baseline":
        teacher_model.load_state_dict(torch.load(os.getcwd() + '/'+args.teacher_model_path, map_location=next(teacher_model.parameters()).device))
        teacher_model.to(device)
        score_mean = args.score_mean
        score_std = args.score_std

    base_env = GymEnv("HalfCheetah-v4", device=device)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )



    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)



    print("normalization constant shape:", env.transform[0].loc.shape)
    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("input_spec:", env.input_spec)
    print("action_spec (as defined by input_spec):", env.action_spec)



    check_env_specs(env)



    rollout = env.rollout(3)
    print("rollout of three steps:", rollout)
    print("Shape of the rollout TensorDict:", rollout.batch_size)



    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        # nn.LazyLinear(num_cells, device=device),
        # nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )



    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )



    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )



    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )



    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))



    # collector = SyncDataCollector(
    #     env,
    #     policy_module,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     split_trajs=False,
    #     device=device,
    # )

    collector = MultiSyncDataCollector(
        create_env_fn=[env for _ in range(8)],
        policy=policy_module,
        total_frames=total_frames,
        frames_per_batch=frames_per_batch,
        split_trajs=False,
        device = device
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )



    advantage_module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )



    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    timesteps_total = 0
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            rewards = deepcopy(tensordict_data['next']['reward'])
            if mode == "llm" or mode == "baseline":
                obs = tensordict_data['observation']
                rm_obs = torch.stack((obs[:,0], obs[:,5]), dim=1).to(device)
                score = teacher_model(rm_obs)
                if mode == "llm":
                    obs_ = tensordict_data['next']['observation']
                    rm_obs_ = torch.stack((obs_[:,0], obs_[:,5]), dim=1).to(device)
                    score_ = teacher_model(rm_obs_)    
                    tensordict_data['next']['reward'] = (score_ - score + 1.0).to(device)
                elif mode == "baseline":
                    # obs_ = tensordict_data['next']['observation']
                    # rm_obs_ = torch.stack((obs_[:,0], obs_[:,5]), dim=1).to(device)
                    # score_ = teacher_model(rm_obs_) 
                    tensordict_data['next']['reward'] = ((score - score_mean) / (score_std+1e-8) + 1.0).to(device)

            advantage_module(tensordict_data)
            
            #print ('tensordict_data', tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

            tensordict_data['next']['reward'] = rewards
            del rewards


        timesteps_total += tensordict_data["step_count"].shape[0]
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = (
            f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        )
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 2 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                episode_reward_mean = eval_rollout["next", "reward"].sum().item()
                logs["eval reward (sum)"].append(
                    episode_reward_mean
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                logs["episode_reward_mean"].append(episode_reward_mean)
                logs["timesteps_total"].append(timesteps_total)
                custom_metric = {
                    "episode_reward_mean": episode_reward_mean,
                    "timesteps_total": timesteps_total
                }
                log_data_to_csv(csv_file_path, custom_metric)
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["timesteps_total"], logs["episode_reward_mean"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["timesteps_total"], logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["timesteps_total"], logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()
    collector.shutdown()


def setup_logging_directory(experiment_name):
    """ Set up logging directories and return the path to the CSV file. """
    base_dir = os.path.join('Experiments', experiment_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # Create a subdirectory named by the current date and time
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, 'AdvisedTrainer_'+current_time)
    os.makedirs(run_dir)
    
    # Path for the CSV file
    csv_file_path = os.path.join(run_dir, 'progress.csv')
    return csv_file_path

def initialize_csv(csv_file_path, fieldnames):
    """ Initialize the CSV file with the column headers. """
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

def log_data_to_csv(csv_file_path, data):
    """ Log data to the CSV file. """
    with open(csv_file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writerow(data)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parser.parse_args()
    print("running with CLS:")
    print(f"name: {args.name}, mode: {args.mode}, teacher model: {args.teacher_model_path}, score mean:{args.score_mean}, score std: {args.score_std}")
    experiment_name = args.name
    csv_file_path = setup_logging_directory(experiment_name)
    initialize_csv(csv_file_path, ['episode_reward_mean', "timesteps_total"])
    
    main(args, csv_file_path)