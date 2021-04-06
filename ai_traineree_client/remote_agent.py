import json
import logging
import requests
import os

from tenacity import retry, stop_after_attempt, after_log
from typing import Any, Dict, List, Tuple, Optional, Union
from .utils import to_list

StateType = List[float]
ActionType = Union[int, List[Union[int, float]]]

SUPPORTED_MODELS = ['dqn', 'ppo', 'ddpg', 'rainbow']

global_logger = logging.getLogger("Retry")


class RemoteAgent:

    name = "RemoteAgent"
    default_url = "https://agents.bar"
    logger = logging.getLogger("RemoteAgent")

    def __init__(self,
        state_size: int, action_size: int,
        agent_model: str, agent_name: str, description: str="", **kwargs):
        """
        An instance of the agent in the Agents Bar.

        Parameters:
            state_size (int): Dimensionality of the state space.
            action_size (int): Dimensionality of the action space.
                In case of discrete space, that's a single dimensions with potential values.
                In case of continious space, that's a number of dimensions in uniform [0, 1] distribution.
            agent_model (str): Name of the model type. Check `ai_traineree_client.SUPPORTED_MODELS`
                for accepted values.
            description (str): Optional. Description for the model, if creating a new one.

        Keyword arguments:
            access_token (str): Default None. Access token to use for authentcation. If none provided
                then one is obtained by logging to the service using credentials.
            username (str): Default None. Overrides username from the env variables.
            password (str): Default None. Overrides password from the env variables.

        """

        super().__init__(**kwargs)

        self.url = self.__parse_url(kwargs)
        access_token = kwargs.pop("access_token", None)
        if access_token is None:
            access_token = self.__login(username=kwargs.pop("username", None), password=kwargs.pop("password", None))
        self.__access_token = access_token
        self._headers = {"Authorization": f"Bearer {self.__access_token}", "accept": "application/json"}

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
        self._discrete = self.agent_model.lower() in ('dqn', 'rainbow')
    
    @staticmethod
    def __parse_url(kwargs) -> str:
        url = kwargs.get("url", RemoteAgent.default_url)
        if url[:4].lower() != "http":
            url = "https://" + url
        return url + "/api/v1"

    def __login(self, username: Optional[str]=None, password: Optional[str]=None):
        username = username if username is not None else os.environ['AGENTS_BAR_USER']
        password = password if password is not None else os.environ['AGENTS_BAR_PASS']
        data = dict(username=username, password=password)
        response = requests.post(f"{self.url}/login/access-token", data=data)
        if response.status_code >= 300:
            self.logger.error(response.text)
            raise ValueError(
                f"Receieved an error while trying to authenticate as username='{username}'." \
                "Please double check your credentials. Error: {response.text}"
            )
        return response.json()['access_token']

    def create_agent(self):
        """Creates a new agent in the service.

        Uses provided information on RemoteAgent instantianation to create a new agent.
        Creating a new agent will fail if the ower already has one with the same name.

        *Note* that it can take a few seconds to create a new agent. In such a case,
        any calls to the agent might fail. To make sure that your program doesn't fail
        either use `ai_traineree_client.wait_until_agent_exists` or manually sleep for
        a few seconds.

        """
        self.logger.debug("Creating an agent (name=%s, model=%s)", self.agent_name, self.agent_model)
        payload = dict(name=self.agent_name, model=self.agent_model, description=self.description, config=self._config)
        response = requests.post(f"{self.url}/agents/", data=json.dumps(payload), headers=self._headers)
        if response.status_code >= 300:
            raise RuntimeError("Unable to create a new agent.\n%s" %response.json())
        return response.json()
    
    def remove(self, *, agent_name: str, quite: bool=True):
        """Deletes the agent.

        **Note** that this action is irreversable. All information about agent will be lost.

        Parameters:
            agent_name (str): You are required to pass the name of the agent as
                              a proof that you're an adult and you know what you're doing.

        """
        if agent_name is None or self.agent_name != agent_name:
            if quite:
                self.logger.warning("You're request for deletion is being ignored. You're welcome.")
                return
            else:
                raise ValueError("You wanted to delete an agent. Are you sure? If so, we need *again* its name.")
        
        self.logger.warning(f"Agent '{agent_name}' is being exterminated")
        response = requests.delete(f"{self.url}/agents/{agent_name}", headers=self._headers)
        
        return response.text

    @property
    def exists(self):
        response = requests.get(f"{self.url}/agents/{self.agent_name}", headers=self._headers)
        return response.status_code == 200

    @staticmethod
    def __validate_agent_model(model):
        if model.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model}' isn't currently supported. Please select one from {SUPPORTED_MODELS}")

    def info(self):
        response = requests.get(f"{self.url}/agents/{self.agent_name}", headers=self._headers)
        return response.json()

    @property
    def hparams(self):
        def make_strings_out_of_things_that_are_not_obvious_numbers(v):
            return str(v) if not isinstance(v, (int, float)) else v
        return {k: make_strings_out_of_things_that_are_not_obvious_numbers(v) for (k, v) in self._config.items()}


    @retry(stop=stop_after_attempt(3), after=after_log(global_logger, logging.INFO))
    def act(self, state, noise: float=0):
        """Asks for action based on provided state.

        Parameters:
            state (List floats): Python list of floats which represent agent's state.
            noise (float): Default 0. Value for epsilon in epsilon-greedy paradigm.
        
        Returns:
            action (a number or list of numbers): Suggested action to take from this state.
                In case of discrete problems this is a single int value. Otwherise it is
                a list of either floats or ints.

        """
        data = json.dumps(state)
        response = requests.post(f"{self.url}/agents/{self.agent_name}/act?noise={noise}", data=data, headers=self._headers)
        if response.status_code >= 400:
            raise RuntimeError(response.text)
        response_json = response.json()
        self.logger.debug("Response: %s", response.text)
        action = response_json['action']
        if self._discrete:
            return int(action[0])
        return action


    @retry(stop=stop_after_attempt(10), after=after_log(global_logger, logging.INFO))
    def step(self, state: StateType, action: ActionType, reward: float, next_state: StateType, done: bool):
        """Providing information from taking a step in environment.

        *Note* that all values have to be python plain values, like ints, floats, lists...
        Unfortunately, numpy, pandas, tensors... aren't currently supported.

        Parameters:
            state (StateType): Current state.
            action (ActionType): Action taken from the current state.
            reward (float): A reward obtained from getting to the next state.
            next_state (StateType): The state that resulted from taking `action` at `state`.
            done (bool): A flag whether the `next_state` is a terminal state. 
        
        """
        step_data = {
            "state": to_list(state), "action": to_list(action), "reward": reward, "next_state": to_list(next_state), "done": done
        }
        data  = {"step_data": step_data}

        response = requests.post(f"{self.url}/agents/{self.agent_name}/step", data=json.dumps(data), headers=self._headers)
        self.logger.debug("Received: %s", response.text)
