# Hopper Torchrl

This is the torchrl implementation codes for the Mujoco Hopper Environment of the paper **A Reward Analysis of Reinforcement Learning from Large Language Model Feedback**.

# Environment Installation
```
conda env create -f environment_hopper_linux.yml
conda activate torchrl
```

# Running Commands

```
python3 hopper_tutorial.py --mode=llm --teacher_model_path=LLM/RM/models/reward_model-llama1k.pth --name try
```

* `--mode`  
  - llm: use the unnormalized potential difference rewards (the default one in the paper)
  - baseline: use the normalized potential difference rewards
  
* `--teacher_model_path` the path to the reward model
* `--name` the name of the logging data file (inside the folder \Experiments)# 1
