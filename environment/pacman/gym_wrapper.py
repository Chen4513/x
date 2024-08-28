import gym

from . import featureExtractors
from . import ghostAgents
from . import layout
from . import pacman
from . import pacmanAgents

import model.model_wrapper as model_wrapper
import model.introspection_model as introspection_model
import teacher_student.fixedAdvise as fixedAdvise


class GymPacman(gym.Env):
    def __init__(self, env_config):
        self.rules = pacman.ClassicGameRules(multiagent = env_config["multiagent"])
        self.rules.quiet = True
        self.num_ghosts = 4
        self.multiagent = env_config["multiagent"]
        self.advice_budget = env_config["advice_budget"]
        self.advice_mode = env_config["advice_mode"]
        self.advice_strategy = env_config["advice_strategy"]
        self.introspection_decay_rate = env_config["introspection_decay_rate"]

        self.layout = layout.getLayout("originalClassic")

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(13,),
            dtype=int
        )

        self.action_info = {
            "action_advice": 0,
            "action_introspection": 0,
            "action_student": 0
        }

        self.game = None
        self.feature_extractor = featureExtractors.VectorFeatureExtractor()

        model_config = env_config["model"]

        self.teacher_model = model_wrapper.ModelWrapper(model_wrapper.ModelType.TORCH)
        self.teacher_model.load(env_config["teacher_model_path"], self.action_space, self.observation_space, model_config)

        self.teacher_dt = model_wrapper.ModelWrapper(model_wrapper.ModelType.TREE)
        self.teacher_dt.load(env_config["teacher_dt_path"], self.action_space, self.observation_space, model_config)

        self.introspection_model = introspection_model.Subtree()
        self.introspection_prob = 1.0

        # benchmark algorithms for comparision
        self.fixed_advise_teacher = fixedAdvise.FixedAdvise(env_config["fixed_advise_type"], self.advice_budget, self.teacher_model)

        self.step_idx = 0
        
    def reset(self):
        if self.multiagent:
            pacman_agent = pacmanAgents.GreedyAgent()
            ghost_agents = [None for _ in range(self.num_ghosts)]
        else:
            ghost_agents = []
            pacman_agent = None
            for idx in range(self.num_ghosts):
                ghost_agents.append(ghostAgents.DirectionalGhost(idx + 1))

        # If there is already a running game, copy over the current introspection probability and budget so it gets carried forward
        if self.game is not None:
            self.advice_budget = self.game.advice_budget
            self.introspection_prob = self.game.introspection_prob

        self.game = self.rules.newGame(
            self.layout,
            pacman_agent,
            ghost_agents,
            self.teacher_model,
            self.teacher_dt,
            self.advice_budget,
            self.advice_mode,
            self.advice_strategy,
            self.introspection_model,
            self.introspection_prob,
            self.introspection_decay_rate,
            self.fixed_advise_teacher)

        features = self.feature_extractor.getUnconditionedFeatures(self.game.state)

        return features

    def step(self, action):
        return self.game.step(action)
