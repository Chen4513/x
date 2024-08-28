from environment.mujoco.half_cheetah_v4 import HalfCheetahEnv

from enum import Enum
import numpy as np
import pickle, time
from PIL import Image
from LLM.apis_h import decode_obs, decode_llm_msg, value_est
from copy import deepcopy
import argparse
import os

class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_llm",
    type=bool,
    default=False,
    choices=[True, False],
    help="training data source"
)
parser.add_argument(
    "--sample_size",
    type=int,
    help="number of sampled state pairs"
)
parser.add_argument(
    "--filename",
    type=str,
    default='llm-ranking_data_train.pkl',
    help="name of the training database"
)
parser.add_argument(
    "--start_id",
    type=int
)

def llm_rank(obs, new_obs, step_idx, base_prompt):

    q = decode_obs(obs, new_obs, step_idx)
    complete_q = base_prompt + q
    print(complete_q)

    while True:

        try:
            llm_val = decode_llm_msg(value_est(complete_q))
        except:
            # pass
            continue
        if llm_val == False:
            # pass
            continue
        break
    # llm_val = False
    return llm_val


def rank_states(obs, new_obs, height_threshold = 0.75):
    # what is our human preference in ranking
    old_height = obs[1]
    new_height = new_obs[1]
    old_vel = obs[5]
    new_vel = new_obs[5]

    old_x = obs[0]
    new_x = new_obs[0]

    if old_height > height_threshold and new_height > height_threshold:
        if new_x > old_x:
            return Ranking.GREATER
        else:
            return Ranking.LESSER
    else:
        if new_x > old_x:
            return Ranking.GREATER
        else:
            return Ranking.LESSER


def collect_ranking_data_llm(env:HalfCheetahEnv, filepath, llm_filepath, sample_size,start_id=0):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    with open(os.getcwd() + '/LLM/prompts/hopper.txt', "r") as file:
        sys_prompt = file.read()

    training_data = {"input": [], "target": [], "llm_target": []}

    input_pairs = data['input']
    
    for step_idx, (obs, new_obs, action) in enumerate(input_pairs):
        print(obs)
        print(action)
        print(new_obs)
        llm_val = llm_rank(obs, new_obs, step_idx, sys_prompt)
        print("======================\n\n")
        training_data["llm_target"].append(llm_val)
        training_data['input'].append([obs, new_obs, action])
        training_data['target'].append(data['target'][step_idx])

        if step_idx % 25 == 0:
            pickle.dump(training_data, open(llm_filepath, "wb"))
            print('----------------------------')
            print('checkpoint! cur length of data file:')
            print(len(training_data["llm_target"]))
        if step_idx >= sample_size:
            pickle.dump(training_data, open(llm_filepath, "wb"))
            print('----------------------------')
            print('checkpoint! cur length of data file:')
            print(len(training_data["llm_target"]))
    
    pickle.dump(training_data, open(llm_filepath, "wb"))
    return training_data


def generate_training_data(env:HalfCheetahEnv, num_samples = 1):
    training_data = {"input": [], "target": [], "llm_target": []}

    for i in range(num_samples):
        obs = env.reset()
        obs = env.randomize_state()
        # action = np.array([env.np_random.uniform(low=-3, high=3)])
        action = env.action_space.sample()
        print(obs)
        print(action)
        env.render()
        new_obs, _, _, _, _ = env.step(action)
        print(new_obs)
        #time.sleep(0.5)
        #env.render()
        #time.sleep(0.5)
        ranking = rank_states(obs, new_obs)
        # print(ranking)
        print("========\n")
        training_data["input"].append([obs, new_obs, action])
        training_data["target"].append(ranking)
    return training_data


def create_training_data(args):
    env = HalfCheetahEnv()
    ''''
    for step_id in range(1000):
         print(step_id * 5)
    '''
    filename = args.filename
    if args.use_llm:
        orifile_name = "hopper_base-v3.pkl"
        data = collect_ranking_data_llm(env, orifile_name, args.filename, args.sample_size, start_id=args.start_id)
    else:
        data = generate_training_data(env, num_samples=args.sample_size)
    pickle.dump(data, open(filename, "wb"))

def validate_training_data(filename="ranking_data_train_2.pkl"):
    training_data = pickle.load(open(filename, "rb"))

    print("{} rows of input and {} rows of output.".format(len(training_data["input"]), len(training_data["target"])))
    
    #print(training_data[agent_idx]["llm_target"])


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.use_llm)
    create_training_data(args)
    validate_training_data(args.filename)