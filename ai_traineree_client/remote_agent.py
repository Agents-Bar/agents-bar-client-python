import json
import logging
import requests

from ai_traineree.agents import AgentBase
from ai_traineree.loggers import DataLogger
from ai_traineree.types import ActionType, StateType, RewardType, DoneType
from typing import Any, Dict, Tuple


SUPPORTED_MODELS = ['dqn', 'ppo', 'ddpg']


class RemoteAgent(AgentBase):

    name = "RemoteAgent"
    logger = logging.getLogger("RemoteAgent")

    def __init__(self, 
        state_size: int, action_size: int,
        url: str, agent_model: str, agent_name: str, description: str="", **kwargs):

        super().__init__(**kwargs)

        self.url = f"http://{url}/api/v1"

        self._config: Dict = {}
        self._config.update(**kwargs)
        self.state_size: int = state_size
        self.action_size: int = action_size
        self._config['state_size'] = state_size
        self._config['action_size'] = action_size
        self.in_features: Tuple[int] = (self.state_size,)
        self.loss: Dict[str, float] = {}


        self.__validate_agent_model(agent_model)
        self.agent_model = agent_model
        self.agent_name = agent_name
        self.description = description

        self.__login()
        self.__create_agent()

    def __login(self):
        data = {'username': 'test@test.test', 'password': 'eloelo350'}
        response = requests.post(f"{self.url}/login/access-token", data=data)

        self.__access_token = response.json()['access_token']
        self._headers = {"Authorization": f"Bearer {self.__access_token}", "accept": "application/json"}

    def __create_agent(self):
        self.logger.debug("Creating agent")
        payload = dict(name=self.agent_name, model=self.agent_model, description=self.description, config=self._config)
        response = requests.post(f"{self.url}/models/", data=json.dumps(payload), headers=self._headers)
    
    @staticmethod
    def __validate_agent_model(model):
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' isn't currently supported. Please select one from {SUPPORTED_MODELS}")

    @property
    def hparams(self):
        def make_strings_out_of_things_that_are_not_obvious_numbers(v):
            return str(v) if not isinstance(v, (int, float)) else v
        return {k: make_strings_out_of_things_that_are_not_obvious_numbers(v) for (k, v) in self._config.items()}


    def _register_param(self, source: Dict[str, Any], name: str, default_value=None, update=False, drop=False) -> Any:
        self._config[name] = value = source.get(name, default_value)
        if drop and name in source:
            del source[name]
        elif update:
            source[name] = value
        return value


    def act(self, state: StateType, noise: float=0):
        data = json.dumps(state.tolist())
        self.logger.debug("Act: %s", data)
        response = requests.post(f"{self.url}/models/{self.agent_name}/act?noise={noise}", data=data, headers=self._headers)
        action = response.json()['action']
        self.logger.debug("Sending action: %s", action)
        if self.agent_model == 'dqn':
            return int(action[0])
        return action


    def step(self, state: StateType, action: ActionType, reward: RewardType, next_state: StateType, done: DoneType):
        list_action = [action] if isinstance(action, (int, float)) else list(action)
        step_data = {"state": state.tolist(), "action": list_action, "reward": reward, "next_state": next_state.tolist(), "done": done}
        data  = {"step_data": step_data}
        self.logger.debug("Step: %s", data)

        response = requests.post(f"{self.url}/models/{self.agent_name}/step", data=json.dumps(data), headers=self._headers)
        self.logger.debug("Received: %s", response.json())


    def log_metrics(self, data_logger: DataLogger, step: int, full_log: bool=False):
        pass

    def save_state(self, path: str):
        """Saves the whole agent state into a local file."""
        pass

    def load_state(self, path: str):
        """Reads the whole agent state from a local file."""
        pass
