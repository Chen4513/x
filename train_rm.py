import torch, csv,os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
#from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import numpy as np
import argparse, os
import torch.nn.functional as F
import pickle
from enum import Enum

from rewardModel import RewardModel
#from ray.rllib.models.torch.misc import SlimConv2d

#AGENT = 0   # {0,1}
LARGE_NETWORK = False
ORI_NETWORK = True
USE_GPT4 = False
USE_STARLING = False
USE_TRUE_DATA = True
USE_LLAMA = True

EVALUATE = True
VALIDATION = not EVALUATE

INIT_LR =  [0.001, 0.01][LARGE_NETWORK]
BATCH_SIZE = 128#[32, 12812768][LARGE_NETWORK]


def make_RM_obs(obs):
    height = obs[0]
    vel = obs[1]
    return np.array([height, vel])


class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    default="train",
    choices=["train", "validate", "tune"],
    help="train or validate"
)

parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the path of the model we need to validate"
)

# Define the Dataset
class RewardDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.obs_data = [[], []]
        self.eq_num = 0

        if USE_LLAMA:
            target = 'target'
        else:
            target = 'target'
        training_data = pickle.load(open(GT_FILE, "rb"))

        for n in range(len(training_data["input"])):
            if str(training_data[target][n]) == str(Ranking.GREATER):
                for d in range(2):
                    self.obs_data[d].append(make_RM_obs(training_data["input"][n][1-d]))
            elif str(training_data[target][n]) == str(Ranking.LESSER) or training_data[target][n] == str(Ranking.EQUAL):
                for d in range(2):
                    self.obs_data[d].append(make_RM_obs(training_data["input"][n][d]))
            else:
                self.eq_num += 1
            if len(self.obs_data[0]) > size:
                break
        print(len(self.obs_data[0]), self.eq_num)
        self.size = min(self.size, len(self.obs_data[0])) - 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = [None,None]
        if idx >= len(self.obs_data[0]):
            #print(f"Index {idx} is out of bounds")
            idx = len(self.obs_data[0]) - 1  # Adjust idx to the last valid index
        for d in range(2):
            '''
            if(idx == 1000):
                break
            
            if idx < len(self.obs_data[d]):
                print(f"Index {idx} out of bounds for obs_data[{d}] with length {len(self.obs_data[d])}")
            return None
            '''
            if not USE_TRUE_DATA:
                data[d] = np.array([float(ob) for ob in self.obs_data[d][idx]]).astype(np.float32)
            else:
                if idx >= len(self.obs_data[d]):
                    #print(f"Index {idx} out of bounds for obs_data[{d}] with length {len(self.obs_data[d])}")
                    raise IndexError(f"Index {idx} out of bounds for obs_data[{d}]")
                #print(idx)
                #print(np.array(self.obs_data).shape)
                data[d] = np.array(self.obs_data[d][idx]).astype(np.float32)
        return torch.tensor(data[0]), torch.tensor(data[1])
    
class ValidDataset(RewardDataset):
    def __init__(self, size):
        self.size = size
        self.obs_data = [[],[]]
        # if EVALUATE:
        #     files = [os.getcwd()+'/data/validating_states_high' + str(AGENT) + '-v2-clean.csv', os.getcwd()+'/data/validating_states_low' + str(AGENT) + '-v2-clean.csv']
        # elif VALIDATION:
            #files = FILES
        
        # training_data = pickle.load(open(GT_FILE, "rb"))
        training_data = pickle.load(open("data/cheetah_latest.pkl", "rb"))
        #print(training_data["input"][0])
        #print(np.array(training_data["input"][0]))

        for n in range(len(training_data["input"])):
            idx = [n, - (n + 1)][EVALUATE]
            if str(training_data["target"][idx]) == str(Ranking.GREATER):
                for d in range(2):
                    self.obs_data[d].append(make_RM_obs(training_data["input"][idx][1-d]))
            elif str(training_data["target"][idx]) == str(Ranking.LESSER):
                for d in range(2):
                    self.obs_data[d].append(make_RM_obs(training_data["input"][idx][d]))
            if len(self.obs_data[0]) > size:
                break

# Custom Loss Function
def custom_loss(r_a, r_b):
    return -nn.functional.logsigmoid(r_a - r_b) #+ 0.001 * 0.5 * (r_a**2 + r_b**2)
    #return 

# Training Function
def train(model, device, data_loader, optimizer, writer, epochs, scheduler):
    model.train()
    global_step = 0
    print(VALI_DATA_SIZE)
    valid_data = ValidDataset(size=VALI_DATA_SIZE)
    valid_loader = DataLoader(valid_data, batch_size=VALI_DATA_SIZE, shuffle=False)
    for epoch in range(epochs):
        total_loss = 0
        print(data_loader)
        for a, b in data_loader:
            a, b = a.to(device), b.to(device)
            optimizer.zero_grad()
            r_a = model(a) # batch_size x 1
            r_b = model(b) # batch_size x 1
            #print(r_a)
            #print(r_b)
            loss = custom_loss(r_a, r_b)
            loss.mean().backward()
            optimizer.step()
            total_loss += loss.sum().item()

            # Log loss to TensorBoard
            if writer != None:
                writer.add_scalar('Training Loss', round(loss.mean().item(),2), global_step)
            global_step += 1

        scheduler.step()
        # Optionally print the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        if args.mode == "train":
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader.dataset)}. Current Learning Rate: {current_lr}")

        # ******** TODO: Fix this
        ave_loss, ave_correct = validate(model, valid_loader, device)
        #ave_loss = 0
        #ave_correct = 0
        if epoch >=5 and ave_correct < 0.1:
            return 100000, 0
        if args.mode == "train":
            print('-------------------------------------------')
        if args.mode == "train":
            torch.save(model.state_dict(), os.getcwd() + '/LLM/RM/models/latestrm.pth')
    return ave_loss, ave_correct

# Define the lambda function for learning rate adjustment
def lr_lambda_selector(lr_list):
    def lr_lambda(epoch):
        if epoch < 50: #180
            return lr_list[0] / INIT_LR #0.0001# Keep the initial learning rate 000133# relu-1fc ori #00004 relu-2fc ori
        elif 50 <= epoch < 70:
            return lr_list[1] / INIT_LR # 00004 relu-1fc ori 00001 relu-2fc ori #ELSE 0.001 LARGE-SPECIAL 0.0006
        else:
            return lr_list[2] / INIT_LR  #0005# 00002 2fc ori 00001 relu-2fc ori
        # if 100 <= epoch < 200: #!
        #     return 0.0008 / INIT_LR 
        # else:
        #     return 0.00000001 / INIT_LR  #!
    return lr_lambda

def validate(model, valid_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    training_ranking_correct_rate = 0
    # No gradient updates needed for validation, so wrap in no_grad to save memory
    with torch.no_grad():
        print(valid_loader)
        for a, b in valid_loader:
            print(a.shape, b.shape)
            a, b = a.to(device), b.to(device)
            r_a = model(a)
            r_b = model(b)

            loss = custom_loss(r_a, r_b)
            total_loss += loss.sum().item()

            training_ranking_correct_rate += len(np.where(r_a.cpu().numpy() > r_b.cpu().numpy())[0])

    # Calculate average loss over all samples
    print(len(valid_loader.dataset))
    avg_loss = total_loss / len(valid_loader.dataset)
    training_ranking_correct_rate /= len(valid_loader.dataset)
    #if training_ranking_correct_rate == 0.0:
    # for r in range(len(r_a)):
    #     print(r_a[r], r_b[r])
    if args.mode != 'tune':
        print(f'Validation Loss: {avg_loss}')
        print('Validation Ranking Correctness:',training_ranking_correct_rate)
    return avg_loss, training_ranking_correct_rate

import itertools

def param_search(param_grid, device, file_path, csv_file_path):
    n = 6
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_correctness = 0.0
    best_params = None
    best_loss = 0.0
    ave_correct = 0.0
    max_correct = 0.0
    ave_loss = 0.0
    
    for param in param_combinations:
        print("*********************")
        if param["batch_size"]*100 < param["dataset_size"]:
            continue
        print(f'Test parameters:\n {param}')
        correctness = []
        loss = []
        print(f'dataset size: {dataset_size}')
        dataset_size = param["dataset_size"]
        batch_size = param["batch_size"]
        dataset = RewardDataset(size=dataset_size)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for i in range(n):
            lr_list = param["lr_list"]
            epoch = param["epoch"]
            model = RewardModel(param).to(device)
            optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
            scheduler = LambdaLR(optimizer, lr_lambda_selector(lr_list=lr_list))
            loss_, correctness_ = train(model, device, data_loader, optimizer, writer=None, epochs=epoch, scheduler=scheduler)
            print(f'in testing iteration: {i}, validation correctness: {correctness_}, validation loss: {loss_}')
            if correctness_ > 0.01:
                loss.append(loss_)
                correctness.append(correctness_)
        ave_correct = np.mean(correctness, keepdims=False)
        if len(correctness) > 0:
            max_correct = np.max(correctness, keepdims=False)
        else:
            max_correct = 0.0
        ave_loss = np.mean(loss, keepdims=False)
        if max_correct > best_correctness:
            best_correctness = max_correct
            best_params = param
            best_loss = ave_loss
        result = param.copy()
        result.update({'validation_loss': ave_loss, 
                       'validation correctness': ave_correct,
                       'validation max correctness': max_correct,
                       "valid trails": len(loss)} )
        append_result_to_json(file_path, result)
        with open(csv_file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            row = [param['batch_size'], param['dataset_size'], param['epoch'], param['lr_list'], param['conv_filters'], param['fc_layer_sizes'],max_correct, ave_correct, ave_loss, len(loss)]
            writer.writerow(row)
        print(f'Tested:\n {param}, \n Validation correctness: {ave_correct}, Validation max correctness: {max_correct}, Validation loss: {ave_loss}, Valid trails: {len(loss)} \n')
        print(f'Best:\n {best_params}, \n Validation max correctness: {best_correctness}, Validation loss: {best_loss} \n\n')
    return best_correctness, best_params, best_loss

import json

def setup_logging_json(file_path):
    # Initialize the file with an empty list
    with open(file_path, 'w') as file:
        json.dump([], file)
    return file_path

def append_result_to_json(file_path, result):
    with open(file_path, 'r+') as file:
        # Read current data from the file
        data = json.load(file)
        # Append the new result
        data.append(result)
        # Reset the file pointer, and overwrite the file
        file.seek(0)
        json.dump(data, file, indent=4)

# Main
if __name__ == "__main__":
    args = parser.parse_args()
    global AGENT, DATASET_SIZE, VALI_DATA_SIZE, FILES

    VALI_DATA_SIZE = 2000#int(DATASET_SIZE / 5)

    if USE_TRUE_DATA:
        GT_FILE = "data/cheetah_latest.pkl"
        # GT_FILE = "ranking_data30k.pkl"
    elif USE_LLAMA:
        GT_FILE = "data/cheetah_latest.pkl"
    
    config = {
        "conv_filters": [],
        "conv_activation": True,
        "fc_layer_sizes": [[2, 256], [256,256],[256,256],[256, 1]],
        "clip_at_last": "",
        "clip_scale": 400,
    }
    lr_list = [4*1e-06, 4*5e-06, 4e-07]
    #lr_list = [0.004, 0.004, 0.00004]
    # lr_list = [0.004, 0.0001, 0.0000001]
    # lr_list = [1e-06, 5e-06, 2e-06]
    epochs = 65
    train_batch_size = 16
    dataset_size = 999
    log_dir = "runs/lr0.004_train"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    if args.mode == 'tune':
        param_grid = {
            "batch_size": [128, 256, 64],
            "epoch": [250, 60, 150],+
            "lr_list": [[0.000001, 0.000005, 0.000002],[4*0.000001, 4*0.000005, 4*0.000002], [0.4*0.000001, 0.4*0.000005, 0.4*0.000002]],
            "dataset_size": [6000, 4000],
            "conv_filters": [
                [
                    [16, [2, 2], 1, (2,1)],
                    ["pool", [2, 2], 2, (0,0)],
                    [32, [2, 2], 1, (0,0)],
                    [64, [2, 2], 1, (0,0)]
                ],
            ],
            "conv_activation": [True],
            "fc_layer_sizes": [*-
                [[256,512],[512,256],[256,128],[128,64],[64,16],[16,1]]
            ],
        }
        file_path = setup_logging_json('small_dataset_search.json')
        csv_file_path = 'small_dataset_search.csv'
        best_correctness, best_params, best_loss = param_search(param_grid, device, file_path, csv_file_path)
        print(f'Best:\n {best_params},\n Validation correctness:\n {best_correctness},\n Validation loss: {best_loss}')
        
    elif args.mode == 'train':
        rate = 0

        while rate == 0:
            model = RewardModel(config).to(device)
            optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
            dataset = RewardDataset(size=dataset_size)
            data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
            # Initialize SummaryWriter for TensorBoard
            writer = None#SummaryWriter(log_dir=log_dir)
            # Initialize the learning rate scheduler
            scheduler = LambdaLR(optimizer, lr_lambda_selector(lr_list=lr_list))
            loss, rate = train(model, device, data_loader, optimizer, writer, epochs=epochs, scheduler=scheduler)

        # Close the SummaryWriter to flush the information to disk
        if writer != None:
            writer.close()
    else:
        model = RewardModel(config).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))

        valid_data = ValidDataset(size=VALI_DATA_SIZE)
        valid_loader = DataLoader(valid_data, batch_size=VALI_DATA_SIZE, shuffle=False)

        validate(model, valid_loader, device)